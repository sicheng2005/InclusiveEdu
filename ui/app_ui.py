import gradio as gr
from agents.router_agent import route_request

def create_ui():
    # 使用 Gradio 构建 Web 界面
    with gr.Blocks(title="无障碍线上教学辅助平台", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 无障碍线上教学辅助平台")
        gr.Markdown("### 针对视障与听障学生的实时课堂交互系统 (Demo版)")
        
        with gr.Row():
            mode_selector = gr.Radio(
                choices=["视障学生端 (课件文字转语音)", "听障学生端 (手语转文字)"],
                label="请选择您的身份",
                value="视障学生端 (课件文字转语音)"
            )
        
        # 视障模式 Tab：老师发文字课件 -> 语音播报给学生
        with gr.Tab("视障学生端 (课件文字转语音)") as tab_voice:
            gr.Markdown("此模式下，系统接收老师发送的课件文字，并实时转换为语音播放给视障学生。")
            teacher_text_input = gr.Textbox(label="老师发送的课件文字", placeholder="例如：同学们好，今天我们学习牛顿第一定律...")
            audio_output = gr.Audio(label="系统语音播报", interactive=False)
            btn_voice = gr.Button("模拟老师发送课件", variant="primary")
            
        # 听障模式 Tab：学生打手语 -> 文字显示给老师
        with gr.Tab("听障学生端 (手语转文字)") as tab_sign:
            gr.Markdown("*注：此处暂时使用文本输入框模拟手语识别模型的输出结果。在完整版中，这里将接入摄像头进行实时手语翻译。*")
            sign_text_input = gr.Textbox(label="手语识别结果 (模拟学生手语输入)", placeholder="例如：老师，我没听懂刚刚的公式。")
            text_output_sign = gr.Textbox(label="老师端屏幕显示的文字", lines=4)
            btn_sign = gr.Button("发送给老师", variant="primary")
            
        # 绑定事件处理函数
        def process_voice_ui(text):
            if not text:
                return None
            # 调用路由智能体
            _, audio_out = route_request("视障学生端 (课件文字转语音)", text)
            return audio_out
            
        btn_voice.click(fn=process_voice_ui, inputs=teacher_text_input, outputs=audio_output)
        
        def process_sign_ui(text):
            if not text:
                return "请输入内容。"
            # 调用路由智能体
            response_text, _ = route_request("听障学生端 (手语转文字)", text)
            return response_text
            
        btn_sign.click(fn=process_sign_ui, inputs=sign_text_input, outputs=text_output_sign)
        
    return app
