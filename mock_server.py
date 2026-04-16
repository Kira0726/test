"""
模拟比赛平台服务器（mock_server.py）

用于在没有真实比赛平台的情况下测试选手程序。

使用方法：
1. 在终端1运行此服务器
2. 在终端2运行 main.py（设置 PLATFORM_URL=http://localhost:8002）
3. 观察日志输出

启动命令：
    python mock_server.py
"""

from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

# 测试任务列表（包含overview信息）
tasks = [
    {
        "task_id": "test_001",
        "target_sla": "Gold",
        "target_reward": 100,
        "overview": {
            "task_id": "test_001",
            "target_sla": "Gold",
            "target_reward": 100
        },
        "messages": [
            {
                "id": "msg_1",
                "eval_request_type": "generate_until",
                "prompt": "请介绍一下人工智能的未来发展趋势。",
                "eval_gen_kwargs": {
                    "max_gen_toks": 100,
                    "temperature": 0.7,
                    "until": ["\n\n", "。"]
                }
            },
            {
                "id": "msg_2",
                "eval_request_type": "loglikelihood",
                "prompt": "中国的首都是",
                "eval_continuation": "北京"
            },
            {
                "id": "msg_3",
                "eval_request_type": "loglikelihood_rolling",
                "prompt": "今天天气很好，适合出门散步。"
            }
        ]
    },
    {
        "task_id": "test_002",
        "target_sla": "Gold",
        "target_reward": 150,
        "overview": {
            "task_id": "test_002",
            "target_sla": "Gold",
            "target_reward": 150
        },
        "messages": [
            {
                "id": "msg_1",
                "eval_request_type": "generate_until",
                "prompt": "解释一下什么是机器学习：",
                "eval_gen_kwargs": {
                    "max_gen_toks": 150,
                    "temperature": 0.5,
                    "until": ["\n\n"]
                }
            },
            {
                "id": "msg_2",
                "eval_request_type": "loglikelihood",
                "prompt": "太阳从",
                "eval_continuation": "东方升起"
            }
        ]
    },
    {
        "task_id": "test_003",
        "target_sla": "Silver",
        "target_reward": 80,
        "overview": {
            "task_id": "test_003",
            "target_sla": "Silver",
            "target_reward": 80
        },
        "messages": [
            {
                "id": "msg_1",
                "eval_request_type": "generate_until",
                "prompt": "1+1等于几？",
                "eval_gen_kwargs": {
                    "max_gen_toks": 20,
                    "temperature": 0.0,
                    "until": ["\n"]
                }
            },
            {
                "id": "msg_2",
                "eval_request_type": "loglikelihood",
                "prompt": "水的化学式是",
                "eval_continuation": "H2O"
            },
            {
                "id": "msg_3",
                "eval_request_type": "loglikelihood_rolling",
                "prompt": "自然科学是研究自然现象和规律的学科。"
            }
        ]
    }
]

task_index = 0


@app.post("/register")
async def register(data: dict):
    """注册队伍"""
    print(f"[平台] 收到注册请求: name={data.get('name')}, token={data.get('token')[:8]}...")
    return {"status": "ok"}


@app.post("/query")
async def query(data: dict):
    """查询任务"""
    global task_index
    print(f"[平台] 收到查询请求: token={data.get('token', '')[:8]}...")
    
    if task_index >= len(tasks):
        print("[平台] 所有任务已发放完毕，返回404")
        raise HTTPException(status_code=404, detail="No more tasks")
    
    task = tasks[task_index]
    print(f"[平台] 返回任务: {task['task_id']}, reward={task['target_reward']}")
    
    return {
        "task_id": task["task_id"],
        "target_sla": task["target_sla"],
        "target_reward": task["target_reward"]
    }


@app.post("/ask")
async def ask(data: dict):
    """接受任务"""
    global task_index
    print(f"[平台] 收到ask请求: task_id={data.get('task_id')}, sla={data.get('sla')}")
    
    if task_index >= len(tasks):
        return {"status": "closed"}
    
    task = tasks[task_index]
    task_id = data.get("task_id")
    
    if task_id != task["task_id"]:
        print(f"[平台] Task ID不匹配")
        return {"status": "rejected", "reason": "Task ID mismatch"}
    
    result = {
        "status": "accepted",
        "task": {
            "overview": task["overview"],
            "messages": task["messages"]
        }
    }
    
    print(f"[平台] ✅ 任务 {task['task_id']} 已接受")
    task_index += 1
    return result


@app.post("/submit")
async def submit(data: dict):
    """提交结果"""
    user = data.get("user", {})
    msg_data = data.get("msg", {})
    messages = msg_data.get("messages", [])
    
    print(f"[平台] 收到提交请求:")
    print(f"  - 队伍: {user.get('name')}")
    print(f"  - messages数量: {len(messages)}")
    
    for i, msg in enumerate(messages):
        eval_type = msg.get("eval_request_type", "unknown")
        response = msg.get("response")
        accuracy = msg.get("accuracy")
        
        if eval_type == "generate_until":
            print(f"  [{i+1}] generate_until: response长度={len(response) if response else 0}")
        elif eval_type in ["loglikelihood", "loglikelihood_rolling"]:
            print(f"  [{i+1}] {eval_type}: accuracy={accuracy:.4f if accuracy else 'None'}")
    
    return {"status": "ok", "message": "Submission received"}


@app.get("/")
async def root():
    """平台状态"""
    return {
        "status": "running",
        "total_tasks": len(tasks),
        "tasks_sent": task_index,
        "tasks_remaining": len(tasks) - task_index
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    print("=" * 60)
    print("  模拟比赛平台服务器")
    print("=" * 60)
    print(f"  共 {len(tasks)} 个测试任务")
    print("  监听地址: http://0.0.0.0:8002")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
