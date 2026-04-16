"""
AI推理挑战赛 - 平台交互客户端（独立版本）

此模块负责与比赛平台的所有交互：
1. 注册队伍
2. 查询任务
3. 接受任务
4. 提交结果

注意：此客户端功能已集成到main.py的platform_worker中，
此文件作为独立模块保留，方便调试和测试。

已修复：
- 注册失败自动重试
- 直接调用推理函数，避免HTTP开销
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any

import httpx

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlatformClient:
    """比赛平台客户端"""
    
    def __init__(
        self,
        platform_url: str,
        token: str,
        team_name: str,
        inference_funcs: Optional[Dict[str, callable]] = None
    ):
        """
        Args:
            platform_url: 平台地址
            token: 认证令牌
            team_name: 队伍名称
            inference_funcs: 推理函数字典 {"generate_until": func, "loglikelihood": func, ...}
        """
        self.platform_url = platform_url.rstrip("/")
        self.token = token
        self.team_name = team_name
        self.inference_funcs = inference_funcs or {}
    
    async def register(self, max_retries: int = 10) -> bool:
        """
        向平台注册队伍（带重试）
        
        Args:
            max_retries: 最大重试次数
        
        Returns:
            注册是否成功
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(max_retries):
                try:
                    resp = await client.post(
                        f"{self.platform_url}/register",
                        json={
                            "name": self.team_name,
                            "token": self.token
                        }
                    )
                    if resp.status_code == 200:
                        logger.info("队伍注册成功")
                        return True
                    logger.warning(f"注册失败 (尝试 {attempt+1}/{max_retries}): {resp.status_code}")
                except Exception as e:
                    logger.warning(f"注册请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
                
                await asyncio.sleep(2)
            
            logger.error("多次注册失败")
            return False
    
    async def query_task(self) -> Optional[Dict[str, Any]]:
        """
        查询可用任务
        
        Returns:
            任务信息字典，无任务时返回None
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{self.platform_url}/query",
                    json={"token": self.token}
                )
                
                if resp.status_code == 404:
                    return None
                
                if resp.status_code == 200:
                    return resp.json()
                
                logger.warning(f"查询任务失败: {resp.status_code}")
                return None
            
            except Exception as e:
                logger.error(f"查询任务请求失败: {e}")
                return None
    
    async def accept_task(self, task_id: int, sla: str) -> Optional[Dict[str, Any]]:
        """
        接受任务
        
        Args:
            task_id: 任务ID
            sla: SLA等级（必须与target_sla完全一致）
        
        Returns:
            接受结果，包含status和task信息
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{self.platform_url}/ask",
                    json={
                        "token": self.token,
                        "task_id": task_id,
                        "sla": sla
                    }
                )
                
                if resp.status_code == 200:
                    result = resp.json()
                    logger.info(f"任务接受结果: {result.get('status')}")
                    return result
                
                logger.warning(f"接受任务失败: {resp.status_code}")
                return None
            
            except Exception as e:
                logger.error(f"接受任务请求失败: {e}")
                return None
    
    async def submit_result(self, task: Dict[str, Any]) -> bool:
        """
        提交任务结果
        
        Args:
            task: 包含overview和messages的完整任务对象
        
        Returns:
            提交是否成功
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    f"{self.platform_url}/submit",
                    json={
                        "user": {
                            "name": self.team_name,
                            "token": self.token
                        },
                        "msg": {
                            "overview": task.get("overview", {}),
                            "messages": task.get("messages", [])
                        }
                    }
                )
                
                if resp.status_code == 200:
                    logger.info("结果提交成功")
                    return True
                else:
                    logger.error(f"提交失败: {resp.status_code} - {resp.text}")
                    return False
            
            except Exception as e:
                logger.error(f"提交请求失败: {e}")
                return False
    
    def call_inference(self, request_type: str, **kwargs) -> Any:
        """
        直接调用本地推理函数（避免HTTP开销）
        
        Args:
            request_type: 推理类型
            **kwargs: 推理参数
        
        Returns:
            推理结果
        """
        if request_type in self.inference_funcs:
            return self.inference_funcs[request_type](**kwargs)
        else:
            logger.warning(f"未知的请求类型: {request_type}")
            return None
    
    async def process_message(self, msg: Dict[str, Any]) -> None:
        """
        处理单条消息（直接调用本地推理函数）
        
        Args:
            msg: 消息对象，会被原地修改
        """
        eval_request_type = msg.get("eval_request_type", "")
        prompt = msg.get("prompt", "")
        
        try:
            if eval_request_type == "generate_until":
                eval_gen_kwargs = msg.get("eval_gen_kwargs", {}) or {}
                
                response = await asyncio.to_thread(
                    self.call_inference,
                    "generate_until",
                    prompt=prompt,
                    max_new_tokens=eval_gen_kwargs.get("max_gen_toks", 256),
                    temperature=eval_gen_kwargs.get("temperature", 0.0),
                    stop_strings=eval_gen_kwargs.get("until", ["\n\n"])
                )
                msg["response"] = response
                msg["accuracy"] = None
            
            elif eval_request_type == "loglikelihood":
                continuation = msg.get("eval_continuation", "")
                logprob = await asyncio.to_thread(
                    self.call_inference,
                    "loglikelihood",
                    prompt=prompt,
                    continuation=continuation
                )
                msg["response"] = None
                msg["accuracy"] = logprob
            
            elif eval_request_type == "loglikelihood_rolling":
                logprob = await asyncio.to_thread(
                    self.call_inference,
                    "loglikelihood_rolling",
                    prompt=prompt
                )
                msg["response"] = None
                msg["accuracy"] = logprob
            
            else:
                logger.warning(f"未知的eval_request_type: {eval_request_type}")
                msg["response"] = None
                msg["accuracy"] = None
        
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            msg["response"] = None
            msg["accuracy"] = None


async def run_worker():
    """运行平台交互工作循环"""
    # 从环境变量读取配置
    platform_url = os.getenv("PLATFORM_URL", "http://localhost:8002")
    token = os.getenv("TOKEN", "")
    team_name = os.getenv("TEAM_NAME", "my_team")
    
    client = PlatformClient(platform_url, token, team_name)
    
    # 注册（带重试）
    if not await client.register():
        logger.error("注册失败，退出")
        return
    
    # 主循环
    while True:
        try:
            # 查询任务
            task_info = await client.query_task()
            
            if task_info is None:
                await asyncio.sleep(0.5)
                continue
            
            task_id = task_info.get("task_id")
            target_sla = task_info.get("target_sla")
            logger.info(f"收到任务: task_id={task_id}, sla={target_sla}")
            
            # 接受任务
            accept_result = await client.accept_task(task_id, target_sla)
            
            if not accept_result:
                await asyncio.sleep(1)
                continue
            
            if accept_result.get("status") != "accepted":
                if accept_result.get("status") == "closed":
                    logger.info("比赛已结束")
                    break
                continue
            
            # 获取任务详情
            task = accept_result.get("task", {})
            messages = task.get("messages", [])
            
            # 处理所有消息
            for msg in messages:
                await client.process_message(msg)
            
            # 提交结果
            await client.submit_result(task)
        
        except Exception as e:
            logger.error(f"处理任务时出错: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_worker())
