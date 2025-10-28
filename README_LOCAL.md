# Multi-Model OCR Table Extraction - Local Deployment

이미지에서 표를 자동으로 추출하는 웹 애플리케이션입니다. **여러 오픈소스 비전-언어 모델**을 사용하여 API 호출 없이 완전히 독립적으로 실행됩니다.

## 주요 특징

✅ **완전한 로컬 실행** - API 키 불필요, 인터넷 연결 불필요 (모델 다운로드 후)
✅ **다중 모델 지원** - DeepSeek VL, MiniCPM-o 2.6 중 선택
✅ **GPU/CPU 지원** - CUDA GPU 또는 CPU에서 실행 가능
✅ **메모리 최적화** - 4bit/8bit quantization 지원
✅ **Docker 지원** - 간편한 배포 및 관리
✅ **유연한 선택** - 속도, 정확도, 메모리 요구사항에 따라 모델 선택 가능

## 시스템 요구사항

### 최소 사양 (1.3B 모델)
- **CPU**: 4코어 이상
- **RAM**: 8GB 이상
- **저장공간**: 10GB 이상
- **OS**: Linux, macOS, Windows (WSL2)

### 권장 사양 (빠른 추론)
- **GPU**: NVIDIA GPU (8GB VRAM 이상)
- **RAM**: 16GB 이상
- **저장공간**: 20GB 이상
- **CUDA**: 11.8 이상

### 7B 모델 사양
- **GPU**: NVIDIA GPU (16GB VRAM 이상) 또는 4bit quantization 사용
- **RAM**: 32GB 이상
- **저장공간**: 30GB 이상

## 설치 방법

### 방법 1: Python 직접 실행 (권장)

#### 1. 저장소 클론

```bash
git clone <repository-url>
cd ocr
```

#### 2. 가상 환경 생성

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### 3. 의존성 설치

```bash
# CPU only
pip install -r requirements.txt

# GPU (CUDA)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. 모델 다운로드

**사용 가능한 모델:**

| 모델 | 크기 | VRAM | 특징 |
|------|------|------|------|
| `deepseek-vl-1.3b` | ~3GB | 4GB | 빠르고 효율적 (기본) |
| `deepseek-vl-7b` | ~15GB | 14GB | 더 정확함 |
| `minicpm-o-2.6` | ~8GB | 8GB | 균형잡힌 성능 |

**다운로드 명령어:**

```bash
# DeepSeek VL 1.3B (기본, 추천)
python download_model.py

# DeepSeek VL 7B (더 정확)
python download_model.py --model-type deepseek-vl-7b

# MiniCPM-o 2.6
python download_model.py --model-type minicpm-o-2.6

# 모든 모델 다운로드
python download_model.py --all
```

모델은 자동으로 `~/.cache/huggingface/` 디렉토리에 다운로드됩니다.

#### 5. 환경 변수 설정 (선택사항)

```bash
cp .env.example .env
```

`.env` 파일에서 모델 선택 및 설정 변경:
```env
# 사용할 모델 선택
MODEL_TYPE=deepseek-vl-1.3b  # 또는 deepseek-vl-7b, minicpm-o-2.6

# 디바이스 설정
DEVICE=auto           # auto, cuda, cpu

# 메모리 최적화
LOAD_IN_8BIT=false    # GPU 메모리 절약 (약간 느림)
LOAD_IN_4BIT=false    # GPU 메모리 더 절약 (더 느림)
```

#### 6. 애플리케이션 실행

```bash
python app_local.py
```

브라우저에서 접속: `http://localhost:8000`

### 방법 2: Docker 실행

Docker를 사용하면 환경 설정이 자동으로 처리됩니다.

#### 1. Docker 이미지 빌드

```bash
docker-compose build
```

#### 2. 컨테이너 실행

```bash
# CPU 모드
docker-compose up

# GPU 모드 (nvidia-docker 필요)
# docker-compose.yml에서 GPU 설정 주석 해제 후
docker-compose up
```

#### 3. 접속

브라우저에서 `http://localhost:8000` 접속

#### Docker 명령어

```bash
# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down

# 중지 및 볼륨 삭제 (모델 캐시 삭제)
docker-compose down -v
```

## 사용 방법

### 1. 이미지 업로드
- "파일 선택" 버튼 클릭 또는
- 드래그 앤 드롭으로 이미지 업로드

### 2. 표 추출
"표 추출하기" 버튼 클릭

### 3. 결과 확인
추출된 표가 구조화된 HTML 테이블로 표시됩니다.

## 성능 최적화

### GPU 메모리 부족 시

**옵션 1: 8bit Quantization**
```bash
# .env 파일
LOAD_IN_8BIT=true
```
- 메모리 사용량: 약 50% 감소
- 속도: 약간 느림 (10-20%)
- 정확도: 거의 동일

**옵션 2: 4bit Quantization**
```bash
# .env 파일
LOAD_IN_4BIT=true
```
- 메모리 사용량: 약 75% 감소
- 속도: 느림 (30-40%)
- 정확도: 약간 감소

**옵션 3: 작은 모델 사용**
```bash
# .env 파일
MODEL_NAME=deepseek-ai/deepseek-vl-1.3b-chat
```

### CPU 실행 시

```bash
# .env 파일
DEVICE=cpu
```

⚠️ CPU 실행은 매우 느립니다 (GPU 대비 10-50배). 테스트용으로만 권장합니다.

### 추론 속도 비교 (참고)

| 구성 | 메모리 | 속도 (이미지당) |
|------|--------|----------------|
| RTX 3090 + 1.3B | 4GB | ~3-5초 |
| RTX 3090 + 7B | 14GB | ~5-10초 |
| RTX 3090 + 7B (8bit) | 7GB | ~8-15초 |
| RTX 3090 + 7B (4bit) | 4GB | ~10-20초 |
| CPU (16 cores) + 1.3B | 6GB | ~30-60초 |

## API 엔드포인트

### `GET /`
웹 인터페이스

### `GET /health`
헬스 체크 및 모델 상태 확인

**응답:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_name": "deepseek-ai/deepseek-vl-1.3b-chat",
    "device": "cuda",
    "cuda_available": true,
    "cuda_device": "NVIDIA GeForce RTX 3090"
}
```

### `POST /api/extract-table`
이미지에서 표 추출

**요청:**
- Content-Type: `multipart/form-data`
- Body: `file` (이미지)

**응답:**
```json
{
    "success": true,
    "filename": "table.png",
    "data": {
        "tables": [
            {
                "table_number": 1,
                "headers": ["이름", "나이", "직업"],
                "rows": [
                    ["홍길동", "30", "개발자"],
                    ["김철수", "25", "디자이너"]
                ],
                "description": "직원 정보"
            }
        ],
        "total_tables": 1
    }
}
```

## 파일 구조

```
ocr/
├── app.py                 # API 버전 (원본)
├── app_local.py          # 로컬 모델 버전 ⭐
├── deepseek_vl.py        # DeepSeek VL 모델 래퍼 ⭐
├── download_model.py     # 모델 다운로드 스크립트 ⭐
├── requirements.txt      # Python 의존성
├── Dockerfile            # Docker 설정 ⭐
├── docker-compose.yml    # Docker Compose 설정 ⭐
├── .env.example          # 환경 변수 템플릿
├── .gitignore
├── README.md             # API 버전 문서
├── README_LOCAL.md       # 로컬 버전 문서 (이 파일) ⭐
└── static/               # 웹 프론트엔드
    ├── index.html
    ├── style.css
    └── script.js
```

## 문제 해결

### 1. 모델 로드 실패

**증상:** "Failed to load model" 에러

**해결:**
```bash
# 모델 다운로드 확인
python download_model.py

# 캐시 디렉토리 확인
ls ~/.cache/huggingface/hub/
```

### 2. CUDA Out of Memory

**증상:** "CUDA out of memory" 에러

**해결:**
```bash
# .env 파일에서
LOAD_IN_8BIT=true
# 또는
LOAD_IN_4BIT=true
```

### 3. 느린 추론 속도

**원인:**
- CPU로 실행 중
- 큰 모델 사용 중
- Quantization 사용 중

**해결:**
- GPU 사용 확인: 헬스 체크에서 `cuda_available: true` 확인
- 작은 모델 사용: `MODEL_NAME=deepseek-ai/deepseek-vl-1.3b-chat`
- Quantization 비활성화

### 4. 표 인식 실패

**원인:**
- 이미지 품질 낮음
- 표 구조 불명확
- 모델이 작음

**해결:**
- 더 선명한 이미지 사용 (최소 1000x1000px)
- 경계선이 명확한 표 사용
- 7B 모델 사용 (더 정확함)

### 5. Docker 내에서 GPU 사용 불가

**증상:** Docker 컨테이너가 GPU를 인식하지 못함

**해결:**
```bash
# nvidia-docker 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# docker-compose.yml에서 GPU 설정 주석 해제
```

## 모델 정보

### DeepSeek VL 1.3B (기본 권장)
- **크기:** ~3GB
- **파라미터:** 1.3B
- **VRAM:** 4GB (fp16), 2GB (8bit), 1.5GB (4bit)
- **속도:** 빠름 ⚡
- **정확도:** 좋음 ⭐⭐⭐
- **추천 용도:** 빠른 프로토타이핑, 실시간 처리

### DeepSeek VL 7B
- **크기:** ~15GB
- **파라미터:** 7B
- **VRAM:** 14GB (fp16), 7GB (8bit), 4GB (4bit)
- **속도:** 중간 🐢
- **정확도:** 매우 좋음 ⭐⭐⭐⭐⭐
- **추천 용도:** 높은 정확도 필요 시, 배치 처리

### MiniCPM-o 2.6 (NEW!)
- **크기:** ~8GB
- **파라미터:** 2.6B
- **VRAM:** 8GB (fp16), 4GB (8bit), 2GB (4bit)
- **속도:** 중상 ⚡⚡
- **정확도:** 좋음 ⭐⭐⭐⭐
- **추천 용도:** 균형잡힌 성능, 범용 사용

### 모델 선택 가이드

| 상황 | 추천 모델 | 이유 |
|------|----------|------|
| GPU 메모리 4GB 이하 | deepseek-vl-1.3b (8bit) | 가장 적은 메모리 사용 |
| GPU 메모리 8GB | minicpm-o-2.6 | 좋은 성능과 효율성 균형 |
| GPU 메모리 16GB 이상 | deepseek-vl-7b | 최고 정확도 |
| 빠른 처리 속도 필요 | deepseek-vl-1.3b | 가장 빠른 추론 |
| 높은 정확도 필요 | deepseek-vl-7b | 가장 정확한 결과 |
| 범용 사용 | minicpm-o-2.6 | 중간 크기, 좋은 성능 |

## 비용 비교

### API 버전 (app.py)
- ✅ 초기 비용 없음
- ✅ 서버 사양 낮아도 됨
- ❌ 사용량당 API 비용 발생
- ❌ 인터넷 연결 필수
- ❌ 데이터 외부 전송

### 로컬 버전 (app_local.py)
- ✅ 사용량 무제한
- ✅ 데이터 외부 전송 없음
- ✅ 인터넷 불필요 (모델 다운로드 후)
- ❌ GPU 서버 필요 (권장)
- ❌ 초기 설치 복잡

## 프로덕션 배포

### 1. systemd 서비스 (Linux)

```bash
# /etc/systemd/system/deepseek-ocr.service
[Unit]
Description=DeepSeek OCR Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/ocr
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python app_local.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable deepseek-ocr
sudo systemctl start deepseek-ocr
```

### 2. Nginx 리버스 프록시

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

### 3. 로드 밸런싱

여러 인스턴스를 실행하여 처리량 증가:

```bash
# 포트 8000, 8001, 8002에서 실행
python app_local.py --port 8000 &
python app_local.py --port 8001 &
python app_local.py --port 8002 &
```

Nginx에서 로드 밸런싱:

```nginx
upstream deepseek_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    location / {
        proxy_pass http://deepseek_backend;
    }
}
```

## 라이선스

MIT License

## 참고 자료

- [DeepSeek VL GitHub](https://github.com/deepseek-ai/DeepSeek-VL)
- [DeepSeek VL HuggingFace](https://huggingface.co/deepseek-ai)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 기여

이슈 및 PR을 환영합니다!

---

Made with DeepSeek VL (Local Inference)
