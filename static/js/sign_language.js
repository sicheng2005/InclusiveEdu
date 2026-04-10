class SignLanguageDetector {
    constructor(videoElement, canvasElement) {
        this.videoElement = videoElement;
        this.canvasElement = canvasElement;
        this.canvasCtx = canvasElement.getContext('2d');
        
        // 冷却时间控制
        this.cooldownTime = 2500; // 识别成功后冷却 2.5 秒
        this.lastDetectTime = 0;
        this.onSignDetected = null;

        // ==========================================
        // 【核心升级】滑动时间窗口 (Sliding Window)
        // 用于存储过去 N 帧的骨骼数据，实现"动态手语"识别
        // ==========================================
        this.history = []; 
        this.windowSize = 25; // 约保存过去 0.8~1 秒的数据 (假设 30fps)
    }

    async init() {
        console.log("正在初始化 MediaPipe Hands 动态时序模型...");
        
        this.hands = new window.Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        this.hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
        });

        this.hands.onResults(this.onResults.bind(this));

        this.camera = new window.Camera(this.videoElement, {
            onFrame: async () => {
                await this.hands.send({image: this.videoElement});
            },
            width: 640,
            height: 480
        });

        this.camera.start();
        console.log("动态手语识别引擎启动成功！");
    }

    onResults(results) {
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        this.canvasCtx.drawImage(results.image, 0, 0, this.canvasElement.width, this.canvasElement.height);

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            // 绘制骨骼
            for (const landmarks of results.multiHandLandmarks) {
                window.drawConnectors(this.canvasCtx, landmarks, window.HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
                window.drawLandmarks(this.canvasCtx, landmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
            }

            // 将当前帧的右手（或第一只手）数据压入历史时间窗口
            const primaryHand = results.multiHandLandmarks[0];
            this.history.push({
                time: Date.now(),
                landmarks: primaryHand
            });

            // 维持滑动窗口大小
            if (this.history.length > this.windowSize) {
                this.history.shift();
            }

            // 执行动态与静态特征分析
            this.analyzeSpatioTemporalFeatures();

        } else {
            // 如果画面中没有手，清空历史轨迹，避免动作断层导致误判
            this.history = [];
        }
        this.canvasCtx.restore();
    }

    // ==========================================
    // 时空特征提取与分析引擎 (Spatio-Temporal Analysis)
    // ==========================================
    analyzeSpatioTemporalFeatures() {
        let currentTime = Date.now();
        if (currentTime - this.lastDetectTime < this.cooldownTime) return;

        // 至少需要 10 帧数据才能判断动态轨迹
        if (this.history.length < 10) return;

        const currentFrame = this.history[this.history.length - 1].landmarks;
        const oldestFrame = this.history[0].landmarks;

        // 1. 提取静态特征 (Static Features) - 当前帧的手指弯曲状态
        const isThumbUp = currentFrame[4].y < currentFrame[3].y && currentFrame[4].y < currentFrame[5].y;
        const isIndexUp = currentFrame[8].y < currentFrame[6].y;
        const isMiddleUp = currentFrame[12].y < currentFrame[10].y;
        const isRingUp = currentFrame[16].y < currentFrame[14].y;
        const isPinkyUp = currentFrame[20].y < currentFrame[18].y;
        const allFingersUp = isThumbUp && isIndexUp && isMiddleUp && isRingUp && isPinkyUp;

        // 2. 提取动态特征 (Dynamic Features) - 计算手腕(0号点)在时间窗口内的位移和方差
        let minX = 1.0, maxX = 0.0, minY = 1.0, maxY = 0.0;
        for (let frame of this.history) {
            let wrist = frame.landmarks[0];
            if (wrist.x < minX) minX = wrist.x;
            if (wrist.x > maxX) maxX = wrist.x;
            if (wrist.y < minY) minY = wrist.y;
            if (wrist.y > maxY) maxY = wrist.y;
        }
        
        const deltaX = maxX - minX; // X轴最大摆动幅度
        const deltaY = maxY - minY; // Y轴最大摆动幅度
        const netMoveY = currentFrame[0].y - oldestFrame[0].y; // Y轴净位移 (正为向下，负为向上)

        // ==========================================
        // 动作规则匹配 (未来可替换为 LSTM/ONNX 模型推理)
        // ==========================================

        // 动作 1：【动态手语 - 没听懂 / 不行】
        // 特征：五指张开 (或大部分张开)，且手掌在 X 轴上有剧烈的左右摆动 (deltaX 较大)，Y 轴相对稳定
        if (isIndexUp && isMiddleUp && isRingUp && deltaX > 0.15 && deltaY < 0.1) {
            this.triggerSign("老师，我没听懂 (左右摆手) 🙅");
            return;
        }

        // 动作 2：【动态手语 - 老师】
        // 特征：食指向上 (其他弯曲)，并且手部有明显的向下移动轨迹 (netMoveY > 0.1)
        if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && netMoveY > 0.1) {
            this.triggerSign("老师好！🧑‍🏫");
            return;
        }

        // 动作 3：【静态手语 - 听懂了 / 赞】
        // 特征：大拇指伸直，其他弯曲，且手部保持相对静止 (deltaX 和 deltaY 都很小)
        if (isThumbUp && !isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && deltaX < 0.05 && deltaY < 0.05) {
            this.triggerSign("老师，我听懂了！👍");
            return;
        }

        // 动作 4：【静态手语 - 提问】
        // 特征：五指全张开，且手部向上举起并保持静止
        if (allFingersUp && deltaX < 0.05 && deltaY < 0.05 && currentFrame[0].y < 0.6) {
            this.triggerSign("老师，我有问题提问 ✋");
            return;
        }

        // 动作 5：【静态手语 - OK / 可以开始了】
        // 特征：食指和大拇指指尖靠近(距离极小)，中指、无名指、小指伸直
        const thumbIndexDist = Math.hypot(currentFrame[4].x - currentFrame[8].x, currentFrame[4].y - currentFrame[8].y);
        if (thumbIndexDist < 0.05 && isMiddleUp && isRingUp && isPinkyUp) {
            this.triggerSign("可以开始了 / OK 👌");
            return;
        }

        /* 
         * 【未来扩展指南：接入深度学习模型】
         * 当你需要支持几百个词汇时，删除上面的 if/else 规则。
         * 将 this.history 转换为一个 shape 为 [windowSize, 21, 3] 的张量 (Tensor)。
         * 使用 onnxruntime-web 加载训练好的 LSTM/Transformer 模型：
         * const tensor = new ort.Tensor('float32', flattenedHistory, [1, 25, 63]);
         * const results = await session.run({ input: tensor });
         * const word = decodeResult(results.output);
         * this.triggerSign(word);
         */
    }

    triggerSign(message) {
        console.log("识别到动态手语：", message);
        this.lastDetectTime = Date.now();
        this.history = []; // 触发后清空历史，避免重复触发粘连
        
        if (this.onSignDetected) {
            this.onSignDetected(message);
        }
    }
}

window.SignLanguageDetector = SignLanguageDetector;