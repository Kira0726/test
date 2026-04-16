"""
AI推理挑战赛 - 选手服务主程序

功能：
1. 延迟加载大语言模型（避免60秒启动超时）
2. 提供HTTP推理接口 (/v1/completions, /v1/loglikelihood, /v1/loglikelihood_rolling)
3. 后台循环与比赛平台交互（注册、查询任务、推理、提交）

比赛配置：
- 队伍名称: 谢谢送我打印机队
- Token: b14902e1f72e561dfb414100c6a3b204
- 平台: http://117.186.102.100/ybai/ubiservice

硬件环境（开发文档）：
- GPU: RTX 5090 x4
- Python: 3.12
- CUDA: 12.8
- 模型: Qwen3-32B

优化：
- 4卡并行加载（device_map="auto"）
- SDPA attention（RTX 5090原生支持，比eager更快）
- TF32 + cuDNN benchmark 加速
- KV Cache加速推理
- bfloat16精度提高稳定性
- 直接调用本地推理函数，避免HTTP开销
- 模型异步加载，支持503状态码
- 注册失败自动重试
- 更好的错误处理和日志
"""

import os
import sys
import asyncio
import logging
from typing import Optional, List

# 设置 HuggingFace 镜像（国内加速）
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

import httpx
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# ============ 无模型模式标志 ============
NO_MODEL_MODE = "--no-model" in sys.argv or os.getenv("NO_MODEL", "0") == "1"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ 比赛配置 ============
# 队伍名称
TEAM_NAME = "谢谢送我打印机队"
# Secret Token
TOKEN = "b14902e1f72e561dfb414100c6a3b204"
# 比赛平台URL
PLATFORM_URL = os.getenv("PLATFORM_URL", "http://117.186.102.100/ybai/ubiservice")
# 模型路径（环境变量优先）
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3-32B")
# 服务端口
PORT = int(os.getenv("CONTESTANT_PORT", "9000"))

# 模型就绪标志
model_ready = False

# 模型和分词器（全局变量）
tokenizer = None
model = None


# ============ 自定义StoppingCriteria ============
class StopOnStrings(StoppingCriteria):
    """在生成时检测停止词并停止"""
    def __init__(self, stop_strings: List[str], tokenizer, check_window: int = 50):
        super().__init__()
        self.stop_strings = [s for s in stop_strings if s]  # 过滤空字符串
        self.tokenizer = tokenizer
        self.check_window = check_window
    
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.check_window or not self.stop_strings:
            return False
        # 检查最近生成的文本
        generated = self.tokenizer.decode(input_ids[0][-self.check_window:])
        for s in self.stop_strings:
            if s in generated:
                return True
        return False


# ============ 数据模型 ============
class GenerateRequest(BaseModel):
    prompt: str
    request_type: str = "generate_until"
    max_tokens: int = 256
    temperature: float = 0.0
    stop: Optional[List[str]] = None


class LoglikelihoodRequest(BaseModel):
    prompt: str
    continuation: Optional[str] = None


# ============ 异步模型加载 ============
async def load_model_async():
    """
    异步加载模型（后台运行，避免阻塞启动）
    
    针对RTX 5090 x4优化：
    - 使用device_map="auto"自动分配到多卡
    - bfloat16精度提高稳定性
    - SDPA attention比eager更快
    - 启用KV Cache加速推理
    - TF32 + cuDNN benchmark加速
    """
    global tokenizer, model, model_ready
    
    # 无模型模式 - 跳过模型加载
    if NO_MODEL_MODE:
        model_ready = True
        logger.info("[无模型模式] 已跳过模型加载")
        return
    
    logger.info(f"开始异步加载模型 from {MODEL_PATH}...")
    logger.info(f"GPU数量: {torch.cuda.device_count()}")
    
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # 确保pad_token存在
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("分词器加载完成，开始加载模型...")
        
        # 加载模型（针对4卡RTX 5090优化）
        # device_map="auto" 自动分配到多卡
        # bfloat16 精度，提高数值稳定性
        # sdpa 使用优化的scaled dot product attention（RTX 5090原生支持）
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",           # 自动分配到多卡（RTX 5090 x4）
            torch_dtype=torch.bfloat16,  # bfloat16精度
            trust_remote_code=True,
            attn_implementation="sdpa"  # RTX 5090优化的attention，比eager更快
        )
        model.eval()
        
        # 启用KV Cache加速推理
        model.config.use_cache = True
        
        # 启用梯度检查点以节省显存（如需要）
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_disable()  # 推理时不需要
        
        # 打印模型分布信息
        if hasattr(model, 'hf_device_map'):
            logger.info(f"模型设备分布: {model.hf_device_map}")
        
        # 设置CUDA优化
        if torch.cuda.is_available():
            # 启用TF32加速（RTX 5090支持，比float32快约3倍）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # cuDNN自动优化
            logger.info("已启用CUDA优化: TF32 + cuDNN benchmark")
        
        model_ready = True
        logger.info("✅ 模型加载完成，服务就绪")
    
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


# ============ 推理核心函数 ============
def generate_until(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    stop_strings: Optional[List[str]] = None
) -> str:
    """
    generate_until: 根据prompt生成文本，支持停止词截断
    
    重要：FAQ规定 - 命中until时，提交的response应去掉停止符本身
    
    优化点：
    - 使用KV Cache加速生成
    - 更好的停止词处理
    - 防止重复生成
    - 正确处理停止符（不包含在结果中）
    """
    global tokenizer, model
    
    # 无模型模式 - 返回模拟响应
    if NO_MODEL_MODE:
        return "[模拟响应] 这是一个测试回答"
    
    # 编码prompt（限制最大长度4K）
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 过滤空字符串
    valid_stop_strings = [s for s in (stop_strings or []) if s]
    
    # 构建生成参数（优化版）
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": max(temperature, 0.01),  # 避免temperature=0导致的问题
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,  # 防止重复生成
    }
    
    # 如果有停止词，添加StoppingCriteria
    if valid_stop_strings:
        stopping_criteria = StoppingCriteriaList([
            StopOnStrings(valid_stop_strings, tokenizer)
        ])
        gen_kwargs["stopping_criteria"] = stopping_criteria
    
    # 执行生成（使用torch.no_grad节省显存）
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # 解码生成的token
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除原始prompt
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    
    # 去除停止符本身（重要：FAQ规定去掉停止符）
    # 但如果模型在停止符之前就停止了，StoppingCriteria可能没触发
    for stop_str in valid_stop_strings:
        if stop_str in generated_text:
            generated_text = generated_text.split(stop_str)[0]
            break  # 找到第一个停止词就停止
    
    return generated_text.strip()


def compute_loglikelihood(prompt: str, continuation: str) -> float:
    """
    loglikelihood: 计算候选答案的对数概率
    
    重要：FAQ规定 - 返回的是"总logprob"，不是平均值
    
    正确的实现：使用拼接法
    1. 将 prompt + continuation 拼接后一起编码
    2. 计算 continuation 部分每个token的对数概率
    3. 返回总logprob（所有token的logprob之和）
    
    优化：更好的边界处理和错误处理
    """
    global tokenizer, model
    
    # 无模型模式 - 返回模拟响应
    if NO_MODEL_MODE:
        return -1.5
    
    # 安全检查
    if not continuation or not continuation.strip():
        return 0.0
    
    # 拼接完整文本（限制长度4K）
    full_text = prompt + continuation
    full_text = full_text[:8192]  # 安全限制
    
    # 编码完整文本
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 分别编码prompt和continuation（不添加特殊token）
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    
    # 计算continuation在完整序列中的位置
    start_pos = len(prompt_ids)
    
    # 安全检查：确保continuation确实在编码范围内
    if start_pos >= inputs["input_ids"].shape[1] or len(cont_ids) == 0:
        return 0.0
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # 取continuation对应位置的logits
    # logits[i]对应的是输入token[i+1]的预测
    # 所以cont[0]对应logits[start_pos-1]
    end_pos = min(start_pos - 1 + len(cont_ids), logits.shape[1])
    logits_for_cont = logits[0, start_pos - 1: end_pos, :]
    
    # 转换为tensor
    actual_cont_len = end_pos - (start_pos - 1)
    if actual_cont_len <= 0:
        return 0.0
    cont_tensor = torch.tensor(cont_ids[:actual_cont_len]).unsqueeze(0).to(model.device)
    
    # 计算log概率并累加（返回总logprob，不是平均值）
    log_probs = torch.nn.functional.log_softmax(logits_for_cont, dim=-1)
    token_logprobs = log_probs.gather(1, cont_tensor.T).squeeze()
    total_logprob = token_logprobs.sum().item()
    
    return total_logprob


def compute_loglikelihood_rolling(prompt: str) -> float:
    """
    loglikelihood_rolling: 计算整段文本的对数概率
    
    重要：FAQ规定 - 返回的是"总logprob"，不是平均值
    """
    global tokenizer, model
    
    # 无模型模式 - 返回模拟响应
    if NO_MODEL_MODE:
        return -2.5
    
    # 安全检查
    if not prompt or not prompt.strip():
        return 0.0
    
    # 编码prompt（限制长度4K）
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 计算log概率
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # 获取token ids
    input_ids = inputs["input_ids"][0]
    
    # 计算每个token的log概率并累加（跳过第一个token）
    # 返回总logprob，不是平均值
    seq_len = min(logits.shape[1] - 1, len(input_ids) - 1)
    if seq_len <= 0:
        return 0.0
    token_logprobs = log_probs[0, :seq_len].gather(1, input_ids[1:seq_len+1].unsqueeze(-1)).squeeze()
    total_logprob = token_logprobs.sum().item()
    
    return total_logprob


# ============ FastAPI应用 ============
app = FastAPI(title="AI Contestant Service")


@app.on_event("startup")
async def startup_event():
    """启动时：立即开始加载模型，启动后台任务"""
    # 后台加载模型（不阻塞启动）
    asyncio.create_task(load_model_async())
    # 启动平台交互worker
    asyncio.create_task(platform_worker())


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "model_ready": model_ready}


@app.post("/v1/completions")
async def completions(request: GenerateRequest):
    """
    文本生成接口
    用于处理 generate_until 类型的请求
    """
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is loading, please retry")
    
    try:
        response_text = await asyncio.to_thread(
            generate_until,
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_strings=request.stop
        )
        return {"response": response_text}
    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/loglikelihood")
async def loglikelihood(request: LoglikelihoodRequest):
    """
    对数概率接口
    用于处理 loglikelihood 类型的请求
    """
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is loading, please retry")
    
    if not request.continuation:
        raise HTTPException(status_code=400, detail="continuation is required")
    
    try:
        logprob = await asyncio.to_thread(
            compute_loglikelihood,
            request.prompt,
            request.continuation
        )
        return {"logprob": logprob}
    except Exception as e:
        logger.error(f"loglikelihood计算失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/loglikelihood_rolling")
async def loglikelihood_rolling(request: LoglikelihoodRequest):
    """
    滚动对数概率接口
    用于处理 loglikelihood_rolling 类型的请求
    """
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is loading, please retry")
    
    try:
        logprob = await asyncio.to_thread(
            compute_loglikelihood_rolling,
            request.prompt
        )
        return {"logprob": logprob}
    except Exception as e:
        logger.error(f"loglikelihood_rolling计算失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 平台交互客户端 ============
async def platform_worker():
    """
    后台任务：与比赛平台交互
    
    比赛规则说明：
    - 查询限制：32次/s
    - 最多持有64个已ask未submit的任务
    - 任务无人领取300s后超时
    - 多个messages的正确性分数会取平均
    - 第一submit有效，后续忽略
    """
    logger.info("🚀 启动平台交互worker...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. 注册（带重试）
        registered = False
        for attempt in range(10):
            try:
                resp = await client.post(
                    f"{PLATFORM_URL}/register",
                    json={"name": TEAM_NAME, "token": TOKEN}
                )
                if resp.status_code == 200:
                    logger.info("✅ 队伍注册成功")
                    registered = True
                    break
                logger.warning(f"注册失败 (尝试 {attempt+1}/10): {resp.status_code}")
                # 打印详细错误信息
                try:
                    error_detail = resp.json()
                    logger.warning(f"错误详情: {error_detail}")
                except:
                    logger.warning(f"响应内容: {resp.text[:500]}")
            except Exception as e:
                logger.warning(f"注册请求失败 (尝试 {attempt+1}/10): {e}")
            
            await asyncio.sleep(2)
        
        if not registered:
            logger.error("❌ 多次注册失败，退出worker")
            return
        
        logger.info("🔄 开始循环查询任务...")
        
        # 2. 循环查询任务
        while True:
            try:
                # 查询任务
                resp = await client.post(
                    f"{PLATFORM_URL}/query",
                    json={"token": TOKEN}
                )
                
                if resp.status_code == 404:
                    # 无任务，等待后重试（避免过快请求，限制32次/s）
                    await asyncio.sleep(0.1)
                    continue
                
                if resp.status_code != 200:
                    logger.warning(f"查询失败: {resp.status_code}")
                    await asyncio.sleep(1)
                    continue
                
                task_info = resp.json()
                task_id = task_info.get("task_id")
                target_sla = task_info.get("target_sla")
                target_reward = task_info.get("target_reward", 0)
                
                logger.info(f"📋 收到任务: task_id={task_id}, sla={target_sla}, reward={target_reward}")
                
                # 等待模型就绪
                while not model_ready:
                    logger.info("⏳ 等待模型加载完成...")
                    await asyncio.sleep(1)
                
                # 3. 接受任务
                ask_resp = await client.post(
                    f"{PLATFORM_URL}/ask",
                    json={
                        "token": TOKEN,
                        "task_id": task_id,
                        "sla": target_sla
                    }
                )
                
                if ask_resp.status_code != 200:
                    logger.warning(f"接受任务失败: {ask_resp.status_code}")
                    continue
                
                ask_data = ask_resp.json()
                if ask_data.get("status") != "accepted":
                    if ask_data.get("status") == "closed":
                        logger.info("🏁 比赛已结束")
                        return
                    logger.warning(f"任务被拒绝: {ask_data}")
                    continue
                
                # 获取完整任务
                task = ask_data.get("task", {})
                messages = task.get("messages", [])
                
                logger.info(f"📝 开始处理 {len(messages)} 条消息...")
                
                # 4. 处理每条message（直接调用推理函数，避免HTTP开销）
                success_count = 0
                fail_count = 0
                
                for msg in messages:
                    eval_request_type = msg.get("eval_request_type", "")
                    prompt = msg.get("prompt", "")
                    
                    try:
                        if eval_request_type == "generate_until":
                            eval_gen_kwargs = msg.get("eval_gen_kwargs", {}) or {}
                            max_gen_toks = eval_gen_kwargs.get("max_gen_toks", 256)
                            temperature = eval_gen_kwargs.get("temperature", 0.0)
                            until = eval_gen_kwargs.get("until", ["\n\n"])
                            
                            # 直接调用推理函数
                            response = await asyncio.to_thread(
                                generate_until,
                                prompt=prompt,
                                max_new_tokens=max_gen_toks,
                                temperature=temperature,
                                stop_strings=until
                            )
                            msg["response"] = response
                            msg["accuracy"] = None
                            success_count += 1
                            logger.debug(f"  ✅ generate_until 完成，响应长度: {len(response)}")
                        
                        elif eval_request_type == "loglikelihood":
                            continuation = msg.get("eval_continuation", "")
                            # 直接调用推理函数（返回总logprob）
                            logprob = await asyncio.to_thread(
                                compute_loglikelihood,
                                prompt=prompt,
                                continuation=continuation
                            )
                            msg["response"] = None
                            msg["accuracy"] = logprob
                            success_count += 1
                            logger.debug(f"  ✅ loglikelihood 完成，logprob: {logprob:.4f}")
                        
                        elif eval_request_type == "loglikelihood_rolling":
                            # 直接调用推理函数（返回总logprob）
                            logprob = await asyncio.to_thread(
                                compute_loglikelihood_rolling,
                                prompt=prompt
                            )
                            msg["response"] = None
                            msg["accuracy"] = logprob
                            success_count += 1
                            logger.debug(f"  ✅ loglikelihood_rolling 完成，logprob: {logprob:.4f}")
                        
                        else:
                            logger.warning(f"  ⚠️ 未知的eval_request_type: {eval_request_type}")
                            msg["response"] = None
                            msg["accuracy"] = None
                            fail_count += 1
                    
                    except Exception as e:
                        logger.error(f"  ❌ 处理message失败: {e}")
                        msg["response"] = None
                        msg["accuracy"] = None
                        fail_count += 1
                
                logger.info(f"📊 任务处理完成: 成功={success_count}, 失败={fail_count}")
                
                # 5. 提交结果（第一submit有效）
                submit_resp = await client.post(
                    f"{PLATFORM_URL}/submit",
                    json={
                        "user": {
                            "name": TEAM_NAME,
                            "token": TOKEN
                        },
                        "msg": {
                            "overview": task.get("overview", {}),
                            "messages": messages
                        }
                    }
                )
                
                if submit_resp.status_code == 200:
                    logger.info(f"✅ 提交成功: task_id={task_id}")
                else:
                    logger.warning(f"⚠️ 提交失败: {submit_resp.status_code} - {submit_resp.text}")
            
            except httpx.TimeoutException:
                logger.warning("请求超时，重试中...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"处理任务时出错: {e}")
                await asyncio.sleep(1)


# ============ 启动入口 ============
if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_available_port(start_port=9000, max_attempts=100):
        """自动查找可用端口"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                    return port
            except OSError:
                continue
        raise RuntimeError("无法找到可用端口")
    
    # 尝试使用配置的端口，如果被占用则自动选择其他端口
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", PORT))
            actual_port = PORT
    except OSError:
        actual_port = find_available_port(PORT + 1)
        print(f"  ⚠️  端口 {PORT} 被占用，自动使用端口 {actual_port}")
    
    print("=" * 60)
    print("  🤖 AI推理挑战赛 - 选手服务")
    print("=" * 60)
    print(f"  队伍名称: {TEAM_NAME}")
    print(f"  平台地址: {PLATFORM_URL}")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  服务端口: {actual_port}")
    print(f"  GPU数量:  {torch.cuda.device_count()}")
    print("=" * 60)
    
    if NO_MODEL_MODE:
        print("  ⚠️  [无模型测试模式] 已启用")
        print("  - 将跳过模型加载")
        print("  - 返回模拟推理结果")
        print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=actual_port)
