def handle_sign_task(sign_text):
    """
    听障辅助模块
    处理流程：学生的手语(模拟为文字) -> 直接传输给老师屏幕显示
    """
    print(f"[Sign Module] 接收到学生的手语翻译文本: {sign_text}")
    
    # 在这个阶段，我们不需要大模型解答，直接将手语翻译的文字格式化后发给老师
    teacher_screen_display = (
        f"【来自听障学生的提问】\n"
        f"{sign_text}"
    )
    
    print("[Sign Module] 已将学生提问发送至老师屏幕。")
    # 听障模式不需要语音输出
    return teacher_screen_display, None
