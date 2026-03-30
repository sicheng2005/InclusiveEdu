from agents.voice_agent import handle_voice_task
from agents.sign_agent import handle_sign_task

def route_request(mode, input_data):
    """
    主控路由 (Router)
    根据用户身份，将数据流向不同的处理模块。
    """
    print(f"[Router] 接收到请求，当前模式: {mode}")
    
    if mode == "视障学生端 (课件文字转语音)":
        return handle_voice_task(input_data)
        
    elif mode == "听障学生端 (手语转文字)":
        return handle_sign_task(input_data)
        
    else:
        return "未知模式，请重新选择。", None
