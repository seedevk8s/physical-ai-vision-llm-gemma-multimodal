"""Hugging Face Gemma API와 대화하는 Summary Memory 챗봇.

최근 대화는 원문으로 유지하고, 오래된 대화는 하나의 누적 요약으로
압축한다. 메모리는 JSON 파일에 저장되므로 프로그램을 다시 실행해도
이전 대화의 핵심 내용을 이어서 사용할 수 있다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


DEFAULT_MODEL_ID = "google/gemma-3-4b-it"
DEFAULT_MEMORY_PATH = "summary_memory.json"
EXIT_COMMANDS = {"exit", "quit", "q", "종료"}


class SummaryMemory:
    """누적 요약과 최근 N턴을 관리한다."""

    def __init__(
        self,
        client: InferenceClient,
        model_id: str,
        memory_path: str | Path,
        recent_turns: int = 2,
        summary_max_tokens: int = 250,
    ) -> None:
        if recent_turns < 1:
            raise ValueError("recent_turns는 1 이상이어야 합니다.")

        self.client = client
        self.model_id = model_id
        self.memory_path = Path(memory_path)
        self.recent_turns = recent_turns
        self.summary_max_tokens = summary_max_tokens
        self.summary = ""
        self.recent_messages: list[dict[str, str]] = []
        self.load()

    def load(self) -> None:
        """기존 메모리 파일이 있으면 불러온다."""
        if not self.memory_path.exists():
            return

        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            self.summary = str(data.get("summary", ""))
            messages = data.get("recent_messages", [])
            if not isinstance(messages, list):
                raise ValueError("recent_messages가 리스트가 아닙니다.")
            self.recent_messages = [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in messages
                if isinstance(item, dict)
                and "role" in item
                and "content" in item
            ]
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
            raise RuntimeError(
                f"메모리 파일을 읽을 수 없습니다: {self.memory_path}"
            ) from error

    def save(self) -> None:
        """현재 메모리를 JSON 파일에 저장한다."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary,
            "recent_messages": self.recent_messages,
        }
        self.memory_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        """요약과 최근 대화를 모두 지운다."""
        self.summary = ""
        self.recent_messages = []
        self.save()

    def build_messages(self, user_input: str) -> list[dict[str, str]]:
        """누적 요약, 최근 대화, 새 질문을 하나의 입력으로 만든다."""
        messages = list(self.recent_messages)
        if self.summary:
            user_content = (
                "[이전 대화 요약]\n"
                f"{self.summary}\n\n"
                "[현재 사용자 메시지]\n"
                f"{user_input}"
            )
        else:
            user_content = user_input

        messages.append({"role": "user", "content": user_content})
        return messages

    def add_turn(self, user_input: str, assistant_answer: str) -> None:
        """새로운 한 턴을 추가하고 오래된 턴을 요약한다."""
        self.recent_messages.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_answer},
            ]
        )
        self._summarize_overflow()
        self.save()

    def _summarize_overflow(self) -> None:
        max_messages = self.recent_turns * 2
        if len(self.recent_messages) <= max_messages:
            return

        # user/assistant 한 쌍 단위로 오래된 대화를 요약한다.
        overflow_count = len(self.recent_messages) - max_messages
        overflow_count -= overflow_count % 2
        old_messages = self.recent_messages[:overflow_count]
        self.recent_messages = self.recent_messages[overflow_count:]

        old_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in old_messages
        )
        previous_summary = self.summary or "(아직 요약 없음)"
        prompt = f"""
다음의 기존 요약과 새로 오래된 대화를 하나의 최신 요약으로 합쳐줘.
사용자의 신상, 선호, 목표, 제약 조건, 결정 사항처럼 이후 대화에 필요한
정보만 보존하고, 중복과 인사말은 제거해. 한국어로 간결하게 작성해.

[기존 요약]
{previous_summary}

[새로 요약할 대화]
{old_text}

[최신 요약]
""".strip()

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.summary_max_tokens,
            temperature=0.2,
        )
        self.summary = extract_answer(response)


def extract_answer(response: Any) -> str:
    """Hugging Face Chat Completion 응답에서 답변을 꺼낸다."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError("모델 응답 형식을 해석할 수 없습니다.") from error
    return str(content).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma Summary Memory 챗봇")
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face 모델 ID"
    )
    parser.add_argument(
        "--memory", default=DEFAULT_MEMORY_PATH, help="메모리 JSON 저장 경로"
    )
    parser.add_argument(
        "--recent-turns",
        type=int,
        default=2,
        help="요약하지 않고 원문으로 유지할 최근 대화 턴 수",
    )
    parser.add_argument("--max-tokens", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(".env 파일에 HF_TOKEN을 설정해 주세요.")

    client = InferenceClient(api_key=token)
    memory = SummaryMemory(
        client=client,
        model_id=args.model_id,
        memory_path=args.memory,
        recent_turns=args.recent_turns,
    )

    print(f"Gemma Summary Memory Chatbot (API: {args.model_id})")
    print("종료: exit / quit / q / 종료")
    print("명령어: /summary (요약 보기), /reset (기억 초기화)\n")

    while True:
        try:
            user_input = input("사용자: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화를 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("대화를 종료합니다.")
            break
        if user_input == "/summary":
            print(f"요약: {memory.summary or '(아직 요약 없음)'}\n")
            continue
        if user_input == "/reset":
            memory.reset()
            print("대화 기억을 초기화했습니다.\n")
            continue

        response = client.chat.completions.create(
            model=args.model_id,
            messages=memory.build_messages(user_input),
            max_tokens=args.max_tokens,
            temperature=0.7,
        )
        answer = extract_answer(response)
        print(f"Gemma: {answer}\n")
        memory.add_turn(user_input, answer)


if __name__ == "__main__":
    main()
