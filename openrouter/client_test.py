"""
ACE-Step OpenRouter API 客户端测试代码

使用 requests 库测试 API 的各个端点和功能模式。

Usage:
    python -m openrouter.test_client
    python -m openrouter.test_client --base-url http://127.0.0.1:8002
    python -m openrouter.test_client --api-key your-api-key
"""

import argparse
import base64
import json
import os
import sys
import time
from typing import Optional

import requests

# This file is an executable API client script, not a pytest test module.
__test__ = False


# =============================================================================
# 配置
# =============================================================================

DEFAULT_BASE_URL = "https://api.acemusic.ai"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")


def get_headers(api_key: Optional[str] = None) -> dict:
    """构建请求头"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def handle_response(resp, audio_filename: str) -> bool:
    """处理 API 响应并保存音频"""
    print(f"状态码: {resp.status_code}")

    if resp.status_code != 200:
        print(f"错误响应: {resp.text}")
        return False

    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    audio_list = message.get("audio") or []

    print(f"\n内容:\n{content if content else '(无文本内容)'}")
    print(f"音频数量: {len(audio_list)}")

    if audio_list and len(audio_list) > 0:
        audio_url = audio_list[0].get("audio_url", {}).get("url", "")
        if audio_url:
            filepath = save_audio(audio_url, audio_filename)
            print(f"音频已保存: {filepath}")
        else:
            print("警告: audio_url 为空")
    else:
        print("警告: 没有返回音频数据")

    return True


def save_audio(audio_url: str, filename: str) -> str:
    """从 base64 data URL 保存音频文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 解析 data URL: data:audio/mpeg;base64,<data>
    if not audio_url.startswith("data:"):
        print(f"  [警告] 无效的 audio URL 格式")
        return ""

    # 提取 base64 数据
    b64_data = audio_url.split(",", 1)[1]
    audio_bytes = base64.b64decode(b64_data)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    return filepath


# =============================================================================
# 测试函数
# =============================================================================

def test_health(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试健康检查端点"""
    print("\n" + "=" * 60)
    print("测试: GET /health")
    print("=" * 60)

    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        print(f"状态码: {resp.status_code}")
        print(f"响应: {json.dumps(resp.text, indent=2, ensure_ascii=False)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_list_models(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试模型列表端点"""
    print("\n" + "=" * 60)
    print("测试: GET /api/v1/models")
    print("=" * 60)

    try:
        resp = requests.get(
            f"{base_url}/api/v1/models",
            headers=get_headers(api_key),
            timeout=10
        )
        print(f"状态码: {resp.status_code}")
        print(f"响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_natural_language_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试自然语言模式 (Sample Mode)"""
    print("\n" + "=" * 60)
    print("测试: 自然语言模式 (Sample Mode)")
    print("=" * 60)

    payload = {
        "messages": [
            {"role": "user", "content": "Generate an upbeat pop song about summer and travel"}
        ],
        "sample_mode": True,
        "audio_config": {
            "vocal_language": "en",
            "duration": 30,
        },
    }

    print(f"请求: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"状态码: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            audio_list = message.get("audio") or []

            print(f"\n内容:\n{content if content else '(无文本内容)'}")
            print(f"音频数量: {len(audio_list)}")

            if audio_list and len(audio_list) > 0:
                audio_item = audio_list[0]
                audio_url = audio_item.get("audio_url", {}).get("url", "")
                if audio_url:
                    filepath = save_audio(audio_url, "test_natural_language.mp3")
                    print(f"音频已保存: {filepath}")
                else:
                    print("警告: audio_url 为空")
            else:
                print("警告: 没有返回音频数据")

            return True
        else:
            print(f"错误响应: {resp.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tagged_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试标签模式 (Tagged Mode)"""
    print("\n" + "=" * 60)
    print("测试: 标签模式 (Tagged Mode)")
    print("=" * 60)

    content = """<prompt>A gentle acoustic ballad in C major, 80 BPM, female vocal</prompt>
<lyrics>[Verse 1]
Sunlight through the window
A brand new day begins

[Chorus]
We are the dreamers
We are the light</lyrics>"""

    payload = {
        "messages": [{"role": "user", "content": content}],
        "audio_config": {
            "vocal_language": "en",
            "duration": 30,
        },
    }

    print(f"请求: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"状态码: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            audio_list = message.get("audio") or []

            print(f"\n内容:\n{content if content else '(无文本内容)'}")
            print(f"音频数量: {len(audio_list)}")

            if audio_list and len(audio_list) > 0:
                audio_url = audio_list[0].get("audio_url", {}).get("url", "")
                if audio_url:
                    filepath = save_audio(audio_url, "test_tagged_mode.mp3")
                    print(f"音频已保存: {filepath}")

            return True
        else:
            print(f"错误响应: {resp.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lyrics_only_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试纯歌词模式 (Lyrics-Only Mode)"""
    print("\n" + "=" * 60)
    print("测试: 纯歌词模式 (Lyrics-Only Mode)")
    print("=" * 60)

    lyrics = """[Verse 1]
Walking down the street
Feeling the beat

[Chorus]
Dance with me tonight
Under the moonlight"""

    payload = {
        "messages": [{"role": "user", "content": lyrics}],
        "audio_config": {
            "vocal_language": "en",
            "duration": 30,
        },
    }

    print(f"请求: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"状态码: {resp.status_code}")

        return handle_response(resp, "test_lyrics_only.mp3")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_instrumental_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试纯音乐模式 (Instrumental Mode)"""
    print("\n" + "=" * 60)
    print("测试: 纯音乐模式 (Instrumental Mode)")
    print("=" * 60)

    payload = {
        "messages": [
            {"role": "user", "content": "<prompt>Epic orchestral cinematic score, dramatic and powerful</prompt>"}
        ],
        "audio_config": {
            "instrumental": True,
            "duration": 30,
        },
    }

    print(f"请求: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"状态码: {resp.status_code}")

        return handle_response(resp, "test_instrumental.mp3")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试流式响应模式 (Streaming Mode)"""
    print("\n" + "=" * 60)
    print("测试: 流式响应模式 (Streaming Mode)")
    print("=" * 60)

    payload = {
        "messages": [
            {"role": "user", "content": "Generate a cheerful guitar piece"}
        ],
        "stream": True,
        "sample_mode": True,
        "audio_config": {
            "instrumental": True,
            "duration": 30,
        },
    }

    print(f"请求: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            stream=True,
            timeout=300
        )
        print(f"状态码: {resp.status_code}")

        if resp.status_code == 200:
            content_parts = []
            audio_url = None

            print("\n接收流式数据:")
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                if line == "data: [DONE]":
                    print("  [DONE]")
                    break

                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    finish_reason = chunk["choices"][0].get("finish_reason")

                    if "role" in delta:
                        print(f"  角色: {delta['role']}")

                    if "content" in delta and delta["content"]:
                        content_parts.append(delta["content"])
                        # 心跳点不打印
                        if delta["content"] != ".":
                            print(f"  内容: {delta['content'][:100]}...")
                        else:
                            print("  [心跳]")

                    if "audio" in delta and delta["audio"]:
                        audio_item = delta["audio"][0]
                        audio_url = audio_item.get("audio_url", {}).get("url", "")
                        if audio_url:
                            print(f"  音频数据已接收 (长度: {len(audio_url)} 字符)")

                    if finish_reason:
                        print(f"  完成原因: {finish_reason}")

                except json.JSONDecodeError as e:
                    print(f"  [解析错误] {e}")

            full_content = "".join(content_parts)
            print(f"\n完整内容:\n{full_content}")

            if audio_url:
                filepath = save_audio(audio_url, "test_streaming.mp3")
                print(f"\n音频已保存: {filepath}")

            return True
        else:
            print(f"错误响应: {resp.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_full_parameters(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试完整参数控制"""
    print("\n" + "=" * 60)
    print("测试: 完整参数控制")
    print("=" * 60)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": "<prompt>Dreamy lo-fi hip hop beat with vinyl crackle</prompt><lyrics>[inst]</lyrics>"
            }
        ],
        "temperature": 0.9,
        "top_p": 0.95,
        "thinking": False,
        "use_cot_caption": True,
        "use_cot_language": False,
        "use_format": True,
        "audio_config": {
            "bpm": 85,
            "duration": 30,
            "instrumental": True,
        },
    }

    print(f"请求: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"状态码: {resp.status_code}")

        return handle_response(resp, "test_full_params.mp3")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling(base_url: str, api_key: Optional[str] = None) -> bool:
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试: 错误处理")
    print("=" * 60)

    # 测试空消息
    print("\n1. 测试空消息:")
    payload = {"messages": []}
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=30
        )
        print(f"  状态码: {resp.status_code}")
        print(f"  响应: {resp.text[:200]}")
    except Exception as e:
        print(f"  错误: {e}")

    # 测试无内容消息
    print("\n2. 测试无内容消息:")
    payload = {"messages": [{"role": "user", "content": ""}]}
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=30
        )
        print(f"  状态码: {resp.status_code}")
        print(f"  响应: {resp.text[:200]}")
    except Exception as e:
        print(f"  错误: {e}")

    return True


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ACE-Step OpenRouter API 客户端测试")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        help=f"API 基础 URL (默认: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="API 密钥 (可选)"
    )
    parser.add_argument(
        "--test",
        choices=[
            "health", "models", "natural", "tagged", "lyrics",
            "instrumental", "streaming", "full", "error", "all"
        ],
        default="health",
        help="要运行的测试 (默认: health)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ACE-Step OpenRouter API 客户端测试")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"API Key: {'已设置' if args.api_key else '未设置'}")
    print(f"输出目录: {OUTPUT_DIR}")

    tests = {
        "health": test_health,
        "models": test_list_models,
        "natural": test_natural_language_mode,
        "tagged": test_tagged_mode,
        "lyrics": test_lyrics_only_mode,
        "instrumental": test_instrumental_mode,
        "streaming": test_streaming_mode,
        "full": test_full_parameters,
        "error": test_error_handling,
    }

    results = {}

    if args.test == "all":
        for name, test_func in tests.items():
            results[name] = test_func(args.base_url, args.api_key)
    else:
        results[args.test] = tests[args.test](args.base_url, args.api_key)

    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    for name, passed in results.items():
        status = "通过" if passed else "失败"
        print(f"  {name}: {status}")

    # 返回退出码
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
