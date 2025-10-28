# DeepSeek OCR Table Extraction Web App

이미지에서 표를 자동으로 추출하는 웹 애플리케이션입니다. DeepSeek의 강력한 비전-언어 모델을 활용하여 이미지 내의 표를 인식하고 구조화된 데이터로 변환합니다.

## 🚀 배포 방식 선택

이 프로젝트는 **두 가지 배포 방식**을 지원합니다:

### ⭐ 로컬 모델 버전 (app_local.py) - 권장!

✅ **완전히 독립적인 실행** - API 키 불필요
✅ **무제한 사용** - 사용량 제한 없음
✅ **데이터 보안** - 모든 처리가 로컬에서 수행
✅ **인터넷 불필요** - 모델 다운로드 후 오프라인 실행 가능
❌ GPU 서버 권장 (CPU도 가능하나 매우 느림)

👉 **[로컬 배포 가이드 보기 (README_LOCAL.md)](./README_LOCAL.md)**

### 📡 API 버전 (app.py) - 간단한 시작

✅ **빠른 시작** - 복잡한 설정 불필요
✅ **낮은 시스템 요구사항** - 일반 서버에서 실행 가능
❌ API 키 필요 (DeepSeek Platform)
❌ 사용량당 비용 발생
❌ 인터넷 연결 필수

👉 **이 문서를 계속 읽어주세요** (아래)

---

> 💡 **권장사항:** 프로덕션 환경이나 대량 처리에는 **로컬 모델 버전**을 사용하세요!
> 빠른 테스트나 프로토타입에는 **API 버전**이 편리합니다.

---

## 주요 기능

- **이미지 업로드**: 드래그 앤 드롭 또는 파일 선택으로 간편한 이미지 업로드
- **자동 표 추출**: DeepSeek OCR API를 사용한 정확한 표 인식 및 추출
- **구조화된 데이터**: 추출된 표를 보기 쉬운 HTML 테이블로 표시
- **다중 표 지원**: 하나의 이미지에서 여러 표 동시 추출
- **반응형 디자인**: 데스크톱과 모바일 기기 모두 지원

## 기술 스택

### Backend
- **FastAPI**: 현대적이고 빠른 Python 웹 프레임워크
- **OpenAI Python SDK**: DeepSeek API 연동 (OpenAI 호환)
- **Pillow**: 이미지 처리
- **Uvicorn**: ASGI 서버

### Frontend
- **Vanilla JavaScript**: 프레임워크 없는 순수 JavaScript
- **HTML5 & CSS3**: 현대적인 웹 표준
- **반응형 디자인**: 모든 기기에서 최적화된 UI

## 설치 방법

### 1. 필수 요구사항

- Python 3.8 이상
- pip (Python 패키지 관리자)
- DeepSeek API 키 ([DeepSeek 플랫폼](https://platform.deepseek.com/)에서 발급)

### 2. 저장소 클론

```bash
git clone <repository-url>
cd ocr
```

### 3. 가상 환경 생성 (권장)

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4. 의존성 설치

```bash
pip install -r requirements.txt
```

### 5. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 API 키를 설정합니다:

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 DeepSeek API 키를 입력합니다:

```env
DEEPSEEK_API_KEY=your_actual_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

**DeepSeek API 키 발급 방법:**
1. [DeepSeek Platform](https://platform.deepseek.com/)에 가입
2. 대시보드에서 API Keys 메뉴로 이동
3. "Create API Key" 클릭하여 새 키 생성
4. 생성된 키를 복사하여 `.env` 파일에 붙여넣기

## 실행 방법

### 개발 모드로 실행

```bash
python app.py
```

또는

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 프로덕션 모드로 실행

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

애플리케이션이 시작되면 브라우저에서 다음 주소로 접속합니다:

```
http://localhost:8000
```

## 사용 방법

### 1. 이미지 업로드

- **방법 1**: "파일 선택" 버튼을 클릭하여 이미지 선택
- **방법 2**: 이미지 파일을 업로드 영역으로 드래그 앤 드롭

### 2. 미리보기 확인

업로드된 이미지의 미리보기가 표시됩니다.

### 3. 표 추출

"표 추출하기" 버튼을 클릭하여 OCR 처리를 시작합니다.

### 4. 결과 확인

추출된 표가 구조화된 형태로 화면에 표시됩니다.

## API 엔드포인트

### `GET /`
메인 웹 페이지를 제공합니다.

### `GET /health`
애플리케이션의 상태를 확인합니다.

**응답 예시:**
```json
{
    "status": "healthy",
    "api_configured": true
}
```

### `POST /api/extract-table`
이미지에서 표를 추출합니다.

**요청:**
- Content-Type: `multipart/form-data`
- Body: `file` (이미지 파일)

**응답 예시:**
```json
{
    "success": true,
    "filename": "table_image.png",
    "data": {
        "tables": [
            {
                "table_number": 1,
                "headers": ["이름", "나이", "직업"],
                "rows": [
                    ["홍길동", "30", "개발자"],
                    ["김철수", "25", "디자이너"]
                ],
                "description": "직원 정보 테이블"
            }
        ],
        "total_tables": 1
    }
}
```

## 지원 이미지 형식

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)

**제한사항:**
- 최대 파일 크기: 10MB
- 권장 해상도: 1000x1000px 이상

## 프로젝트 구조

```
ocr/
├── app.py                 # FastAPI 백엔드 애플리케이션
├── requirements.txt       # Python 의존성
├── .env                   # 환경 변수 (git에 포함되지 않음)
├── .env.example          # 환경 변수 템플릿
├── .gitignore            # Git 무시 파일 목록
├── README.md             # 프로젝트 문서
└── static/               # 정적 파일
    ├── index.html        # 메인 HTML 페이지
    ├── style.css         # CSS 스타일시트
    └── script.js         # JavaScript 로직
```

## 문제 해결

### API 키 오류

**증상:** "DeepSeek API key not configured" 오류 발생

**해결 방법:**
1. `.env` 파일이 프로젝트 루트 디렉토리에 있는지 확인
2. `.env` 파일에 올바른 API 키가 설정되어 있는지 확인
3. 애플리케이션을 재시작

### 이미지 업로드 오류

**증상:** "Invalid file type" 오류 발생

**해결 방법:**
- 지원되는 이미지 형식인지 확인 (PNG, JPEG, GIF, BMP, WebP)
- 파일 크기가 10MB 이하인지 확인

### 표 추출 실패

**증상:** "No tables detected" 메시지 표시

**가능한 원인:**
1. 이미지 품질이 낮음 → 더 선명한 이미지 사용
2. 표의 구조가 명확하지 않음 → 경계선이 명확한 표 사용
3. 이미지 해상도가 낮음 → 더 높은 해상도의 이미지 사용

## 개발 가이드

### 로컬 개발 환경 설정

```bash
# 개발 모드로 실행 (자동 리로드 활성화)
uvicorn app:app --reload

# 디버그 로그 활성화
uvicorn app:app --reload --log-level debug
```

### 코드 수정 시 주의사항

1. **app.py**: 백엔드 로직 수정
   - DeepSeek API 호출 설정은 `extract_table_with_deepseek()` 함수에서 수정
   - 프롬프트 수정으로 추출 정확도 향상 가능

2. **static/script.js**: 프론트엔드 로직 수정
   - 테이블 렌더링 로직은 `createTableHtml()` 함수에서 수정

3. **static/style.css**: UI 스타일 수정
   - CSS 변수를 통한 일관된 디자인 유지

## 성능 최적화

### 프로덕션 배포 시 권장 설정

```bash
# 워커 프로세스 수 증가 (CPU 코어 수에 맞게 조정)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# 또는 Gunicorn 사용
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 캐싱 및 최적화

- 정적 파일은 CDN을 통해 제공 권장
- 대용량 이미지는 클라이언트 사이드에서 압축 후 전송
- API 응답 캐싱 고려 (동일 이미지 재처리 방지)

## 라이선스

MIT License

## 기여

프로젝트에 기여하고 싶으시다면:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 문의 및 지원

- 이슈가 있으시면 GitHub Issues를 통해 제보해주세요
- 기능 제안은 Pull Request로 제출해주세요

## 참고 자료

- [DeepSeek API Documentation](https://platform.deepseek.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

---

Made with DeepSeek OCR API
