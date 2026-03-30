from core_models.tts_model import text_to_speech

def handle_voice_task(teacher_text):
    """
    视障辅助模块
    处理流程：老师的课件文字 -> TTS -> 语音播放给学生
    """
    print(f"[Voice Module] 正在处理老师的课件内容: {teacher_text}")
    
    # 1. 文字转语音 (TTS)
    audio_output_path = text_to_speech(teacher_text)
    
    print("[Voice Module] 课件已转换为语音。")
    # 返回文字（可选）和生成的音频路径
    return teacher_text, audio_output_path
