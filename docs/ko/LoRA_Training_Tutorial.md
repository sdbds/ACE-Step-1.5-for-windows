# ACE-Step 1.5 LoRA 학습 튜토리얼

## 하드웨어 요구사항

| VRAM | 설명 |
|------|------|
| 16 GB (최소) | 일반적으로 사용 가능하나, 긴 곡의 경우 메모리 부족이 발생할 수 있습니다 |
| 20 GB 이상 (권장) | 전체 길이의 곡을 처리 가능. 학습 중 VRAM 사용량은 보통 17 GB 수준입니다 |

> **참고:** 학습 시작 전 전처리 단계에서 VRAM을 확보하기 위해 Gradio를 여러 번 재시작해야 할 수 있습니다. 구체적인 시점은 이후 단계에서 안내합니다.

## 면책 조항

본 튜토리얼은 **나유탄성인 (NayutalieN)** 의 앨범 *ナユタン星からの物体Y* (총 13곡)을 데모로 사용하며, 500 에포크(배치 사이즈 1)로 학습했습니다. **본 튜토리얼은 LoRA 파인튜닝 기술을 이해하기 위한 교육 목적으로만 사용됩니다. 자신의 원작으로 LoRA를 학습해 주세요.**

개발자로서 나유탄성인의 작품을 매우 좋아하여 앨범 하나를 예시로 선택했습니다. 권리 보유자분께서 본 튜토리얼이 합법적인 권리를 침해한다고 판단하시면 즉시 연락 주세요. 유효한 통지를 받은 후 관련 콘텐츠를 삭제하겠습니다.

기술은 합리적이고 합법적으로 사용되어야 합니다. 아티스트의 창작물을 존중하고, 원작 아티스트의 명예, 권리 또는 이익을 **손상시키거나 해치는** 행위를 하지 마세요.

---

## 데이터 준비

> **팁:** 프로그래밍에 익숙하지 않은 경우, 이 문서를 Claude Code / Codex CLI / Cursor / Copilot 등의 AI 코딩 도구에 전달하여 작업을 대신 수행하게 할 수 있습니다.

### 개요

각 곡의 학습 데이터는 다음 항목으로 구성됩니다:

1. **오디오 파일** — `.mp3`, `.wav`, `.flac`, `.ogg`, `.opus` 형식 지원
2. **가사** — 오디오와 동일한 이름의 `.lyrics.txt` 파일 (하위 호환을 위해 `.txt`도 지원)
3. **어노테이션 데이터** — `caption`, `bpm`, `keyscale`, `timesignature`, `language` 등의 메타데이터

### 어노테이션 데이터 형식

완전한 어노테이션 데이터를 보유하고 있다면, JSON 파일을 생성하여 오디오 및 가사와 같은 디렉토리에 배치할 수 있습니다. 파일 구조는 다음과 같습니다:

```
dataset/
├── song1.mp3               # 오디오
├── song1.lyrics.txt        # 가사
├── song1.json              # 어노테이션 (선택)
├── song1.caption.txt       # 캡션 (선택, JSON에 포함할 수도 있음)
├── song2.mp3
├── song2.lyrics.txt
├── song2.json
└── ...
```

JSON 파일 구조 (모든 필드는 선택 사항):

```json
{
    "caption": "A high-energy J-pop track with synthesizer leads and fast tempo",
    "bpm": 190,
    "keyscale": "D major",
    "timesignature": "4",
    "language": "ja"
}
```

어노테이션 데이터가 없는 경우, 이후 섹션에서 소개하는 방법으로 취득할 수 있습니다.

---

### 가사

가사를 오디오 파일과 동일한 이름의 `.lyrics.txt` 파일로 저장하고 같은 디렉토리에 배치하세요. 가사의 정확성을 확인해 주세요.

스캔 시 가사 파일 검색 우선순위:

1. `{파일명}.lyrics.txt` (권장)
2. `{파일명}.txt` (하위 호환)

#### 가사 전사

기존 가사 텍스트가 없는 경우, 다음 도구를 사용하여 전사할 수 있습니다:

| 도구 | 구조화 태그 | 정확도 | 사용 난이도 | 배포 방식 |
|------|-----------|--------|-----------|----------|
| [acestep-transcriber](https://huggingface.co/ACE-Step/acestep-transcriber) | 없음 | 오류 가능성 있음 | 높음 (모델 배포 필요) | 자체 호스팅 |
| [Gemini](https://aistudio.google.com/) | 있음 | 오류 가능성 있음 | 낮음 | 유료 API |
| [Whisper](https://github.com/openai/whisper) | 없음 | 오류 가능성 있음 | 보통 | 자체 호스팅 / 유료 API |
| [ElevenLabs](https://elevenlabs.io/app/developers) | 없음 | 오류 가능성 있음 | 보통 | 유료 API (무료 크레딧 제공) |

본 프로젝트는 `scripts/lora_data_prepare/`에 해당 전사 스크립트를 제공합니다:

- `whisper_transcription.py` — OpenAI Whisper API를 통한 전사
- `elevenlabs_transcription.py` — ElevenLabs Scribe API를 통한 전사

두 스크립트 모두 `process_folder()` 메서드를 통한 폴더 일괄 처리를 지원합니다.

#### 검토 및 정제 (필수)

전사된 가사에는 오류가 포함될 수 있으며, **반드시 수동으로 검토하고 수정해야 합니다**.

LRC 형식의 가사를 사용하는 경우, 타임스탬프를 제거해야 합니다. 다음은 간단한 정제 예시입니다:

```python
import re

def clean_lrc_content(lines):
    """LRC 파일 내용을 정제하고 타임스탬프를 제거"""
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 타임스탬프 제거 [mm:ss.x] [mm:ss.xx] [mm:ss.xxx]
        cleaned = re.sub(r"\[\d{2}:\d{2}\.\d{1,3}\]", "", line)
        result.append(cleaned)

    # 끝부분 빈 줄 제거
    while result and not result[-1]:
        result.pop()

    return result
```

#### 구조화 태그 (선택)

가사에 구조화 태그(`[Verse]`, `[Chorus]` 등)가 포함되어 있으면, 모델이 곡의 구조를 더 효과적으로 학습할 수 있습니다. 구조화 태그 없이도 정상적으로 학습이 가능합니다.

> **팁:** [Gemini](https://aistudio.google.com/)를 사용하여 기존 가사에 구조화 태그를 추가할 수 있습니다.

예시:

```
[Intro]
La la la...

[Verse 1]
Walking down the empty street
Echoes dancing at my feet

[Chorus]
We are the stars tonight
Shining through the endless sky

[Bridge]
Close your eyes and feel the sound
```

---

### 자동 어노테이션

#### 1. BPM 및 Key 취득

[Key-BPM-Finder](https://vocalremover.org/key-bpm-finder)를 사용하여 BPM과 키 어노테이션을 온라인으로 취득합니다:

1. 웹 페이지를 열고 **Browse my files**를 클릭하여 처리할 오디오 파일을 선택합니다 (한 번에 너무 많이 처리하면 멈출 수 있으므로, 분할 처리 후 CSV를 병합하는 것을 권장합니다). 처리는 로컬에서 수행되며 서버에 업로드되지 않습니다.
   ![key-bpm-finder-0.jpg](../pics/key-bpm-finder-0.jpg)

2. 처리 완료 후, **Export CSV**를 클릭하여 CSV 파일을 다운로드합니다.
   ![key-bpm-finder-1.jpg](../pics/key-bpm-finder-1.jpg)

3. CSV 파일 내용 예시:

   ```csv
   File,Artist,Title,BPM,Key,Camelot
   song1.wav,,,190,D major,10B
   song2.wav,,,128,A minor,8A
   ```

4. CSV 파일을 데이터셋 폴더에 배치합니다. 캡션 데이터를 추가하려면 `Camelot` 열 뒤에 새 열을 추가하세요.

#### 2. Caption 취득

다음 방법으로 곡의 캡션을 취득할 수 있습니다:

- **acestep-5Hz-lm 사용** (0.6B / 1.7B / 4B) — Gradio UI의 Auto Label 기능에서 호출 (이후 단계 참조)
- **Gemini API 사용** — 스크립트 `scripts/lora_data_prepare/gemini_caption.py`를 참조. `process_folder()`로 일괄 처리를 지원하며, 각 오디오 파일에 대해 다음을 생성합니다:
  - `{파일명}.lyrics.txt` — 가사
  - `{파일명}.caption.txt` — 캡션 설명

---

## 데이터 전처리

데이터가 준비되면 Gradio UI를 사용하여 데이터 검토 및 전처리를 수행합니다.

> **중요:** 시작 스크립트를 사용하는 경우, 서비스 사전 초기화를 비활성화하도록 시작 매개변수를 수정해야 합니다:
>
> - **Windows** (`start_gradio_ui.bat`): `if not defined INIT_SERVICE set INIT_SERVICE=--init_service true`를 `if not defined INIT_SERVICE set INIT_SERVICE=--init_service false`로 변경
> - **Linux/macOS** (`start_gradio_ui.sh`): `: "${INIT_SERVICE:=--init_service true}"`를 `: "${INIT_SERVICE:=--init_service false}"`로 변경

Gradio UI를 시작합니다 (시작 스크립트 또는 `acestep/acestep_v15_pipeline.py` 직접 실행).

### 단계 1: 모델 로드

- **LM으로 캡션을 생성해야 하는 경우:** 초기화 시 사용할 LM 모델(acestep-5Hz-lm-0.6B / 1.7B / 4B)을 선택합니다.
  ![](../pics/00_select_model_to_load.jpg)

- **LM이 필요하지 않은 경우:** LM 모델을 선택하지 마세요.
  ![](../pics/00_select_model_to_load_1.jpg)

### 단계 2: 데이터 로드

**LoRA Training** 탭으로 전환하고, 데이터셋 디렉토리 경로를 입력한 후 **Scan**을 클릭합니다.

스캐너는 다음 파일을 자동으로 인식합니다:

| 파일 | 설명 |
|------|------|
| `*.mp3` / `*.wav` / `*.flac` / ... | 오디오 파일 |
| `{파일명}.lyrics.txt` (또는 `{파일명}.txt`) | 가사 |
| `{파일명}.caption.txt` | 캡션 설명 |
| `{파일명}.json` | 어노테이션 메타데이터 (caption / bpm / keyscale / timesignature / language) |
| `*.csv` | BPM / Key 일괄 어노테이션 (Key-BPM-Finder에서 내보내기) |

![](../pics/01_load_dataset_path.jpg)

### 단계 3: 데이터셋 미리보기 및 조정

- **Duration** — 오디오 파일에서 자동으로 읽기
- **Lyrics** — 동일한 이름의 `.lyrics.txt` 파일이 필요 (`.txt`도 지원)
- **Labeled** — 캡션이 있으면 ✅, 없으면 ❌로 표시
- **BPM / Key / Caption** — JSON 또는 CSV 파일에서 로드
- 데이터셋이 모두 인스트루멘탈이 아닌 경우, **All Instrumental** 체크를 해제하세요
- **Format Lyrics** 및 **Transcribe Lyrics** 기능은 현재 비활성화 상태입니다 ([acestep-transcriber](https://huggingface.co/ACE-Step/acestep-transcriber) 미연동으로 인해 LM 직접 사용 시 환각 발생 가능)
- **Custom Trigger Tag**를 입력하세요 (현재 효과가 제한적이며, `Replace Caption` 이외의 옵션이면 괜찮습니다)
- **Genre Ratio**는 캡션 대신 장르를 사용하는 샘플 비율을 제어합니다. 현재 LM이 생성하는 장르 설명은 캡션에 비해 부족하므로 0으로 유지하세요

![](../pics/02_preview_dataset.jpg)

### 단계 4: Auto Label Data

- 이미 캡션이 있는 경우, 이 단계를 건너뛸 수 있습니다
- 데이터에 캡션이 없는 경우, LM 추론을 통해 생성할 수 있습니다
- BPM / Key 값이 없는 경우, 먼저 [Key-BPM-Finder](https://vocalremover.org/key-bpm-finder)로 취득하세요. LM으로 직접 생성하면 환각이 발생합니다

![](../pics/03_label_data.jpg)

### 단계 5: 데이터 미리보기 및 편집

필요한 경우, 데이터를 항목별로 검토하고 수정할 수 있습니다. **각 데이터 편집 후 반드시 저장을 클릭하세요.**

![](../pics/04_edit_data.jpg)

### 단계 6: 데이터셋 저장

저장 경로를 입력하고 데이터셋을 JSON 파일로 저장합니다.

![](../pics/05_save_dataset.jpg)

### 단계 7: 전처리를 통한 Tensor 파일 생성

> **주의:** 이전에 LM으로 캡션을 생성했고 VRAM이 부족한 경우, 먼저 Gradio를 재시작하여 VRAM을 확보하세요. 재시작 시 **LM 모델을 선택하지 마세요**. ��시작 후, 저장된 JSON 파일의 경로를 입력하고 로드합니다.

Tensor 파일 저장 경로를 입력하고 전처리를 시작한 후 완료를 기다립니다.

![](../pics/06_preprocess_tensor.jpg)

---

## 학습

> **주의:** Tensor 파일 생성 후에도 VRAM을 확보하기 위해 Gradio를 재시작하는 것을 권장합니다.

1. **Train LoRA** 탭으로 전환하고, Tensor 파일 경로를 입력하여 데이터셋을 로드합니다.
2. 학습 파라미터에 익숙하지 않은 경우, 기본값을 사용해도 됩니다.

### 파라미터 참고

| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| **Max Epochs** | 데이터셋 크기에 따라 조정 | 약 100곡 → 500 에포크; 10–20곡 → 800 에포크 (참고용) |
| **Batch Size** | VRAM이 충분하면 증가 가능 | 1 (기본값), VRAM이 충분하면 2 또는 4 |
| **Save Every N Epochs** | 체크포인트 저장 간격 | Max Epochs가 작으면 짧게, 크면 길게 설정 |

> 위 수치는 참고용입니다. 실제 상황에 맞게 조정해 주세요.

3. **Start Training**을 클릭하고 학습 완료를 기다립니다.

![](../pics/07_train.jpg)

---

## LoRA 사용

1. 학습 완료 후 **Gradio를 재시작**하고 모델을 다시 로드합니다 (LM 모델은 선택하지 마세요).
2. 모델 초기화 완료 후, 학습된 LoRA 가중치를 로드합니다.
   ![](../pics/08_load_lora.jpg)
3. 음악 생성을 시작합니다.

축하합니다! LoRA 학습의 전체 과정을 완료했습니다.
