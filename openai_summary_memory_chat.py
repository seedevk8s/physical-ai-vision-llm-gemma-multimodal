"""OpenAI Responses API를 사용하는 Summary Memory 챗봇.

최근 N턴은 원문으로 유지하고 오래된 대화는 하나의 누적 요약으로
압축한다. 요약과 최근 대화는 JSON 파일에 저장되어 재실행 후에도
이어진다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_MEMORY_PATH = "openai_summary_memory.json"
EXIT_COMMANDS = {"exit", "quit", "q", "종료"}


class SummaryMemory:
    """OpenAI API를 이용해 누적 요약과 최근 대화를 관리한다."""

    def __init__(
        self,
        client: Any,
        model: str,
        memory_path: str | Path,
        recent_turns: int = 2,
        summary_max_tokens: int = 250,
    ) -> None:
        if recent_turns < 1:
            raise ValueError("recent_turns는 1 이상이어야 합니다.")

        self.client = client
        self.model = model
        self.memory_path = Path(memory_path)
        self.recent_turns = recent_turns
        self.summary_max_tokens = summary_max_tokens
        self.summary = ""
        self.recent_messages: list[dict[str, str]] = []
        self.load()

    def load(self) -> None:
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
                and item.get("role") in {"user", "assistant"}
                and "content" in item
            ]
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
            raise RuntimeError(
                f"메모리 파일을 읽을 수 없습니다: {self.memory_path}"
            ) from error

    def save(self) -> None:
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
        self.summary = ""
        self.recent_messages = []
        self.save()

    def build_input(self, user_input: str) -> list[dict[str, str]]:
        """누적 요약, 최근 대화, 현재 질문을 Responses 입력으로 만든다."""
        messages = list(self.recent_messages)
        if self.summary:
            current_message = (
                "[이전 대화 요약]\n"
                f"{self.summary}\n\n"
                "[현재 사용자 메시지]\n"
                f"{user_input}"
            )
        else:
            current_message = user_input

        messages.append({"role": "user", "content": current_message})
        return messages

    def add_turn(self, user_input: str, assistant_answer: str) -> None:
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

        overflow_count = len(self.recent_messages) - max_messages
        overflow_count -= overflow_count % 2
        old_messages = self.recent_messages[:overflow_count]
        old_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in old_messages
        )
        previous_summary = self.summary or "(아직 요약 없음)"
        summary_prompt = f"""
다음의 기존 요약과 새로 오래된 대화를 하나의 최신 요약으로 합쳐줘.
사용자의 신상, 선호, 목표, 제약 조건, 결정 사항처럼 이후 대화에 필요한
정보만 보존하고 중복과 인사말은 제거해. 한국어로 간결하게 작성해.

[기존 요약]
{previous_summary}

[새로 요약할 대화]
{old_text}

[최신 요약]
""".strip()

        # 요약 호출이 성공한 뒤에만 원본 메시지를 제거한다.
        response = self.client.responses.create(
            model=self.model,
            input=summary_prompt,
            max_output_tokens=self.summary_max_tokens,
            store=False,
        )
        self.summary = extract_answer(response)
        self.recent_messages = self.recent_messages[overflow_count:]


def extract_answer(response: Any) -> str:
    answer = getattr(response, "output_text", None)
    if not answer or not str(answer).strip():
        raise RuntimeError("OpenAI 응답에서 텍스트를 찾을 수 없습니다.")
    return str(answer).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI Summary Memory 챗봇")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI 모델 ID")
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

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "openai와 python-dotenv가 필요합니다. "
            "pip install openai python-dotenv 를 실행해 주세요."
        ) from error

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")

    client = OpenAI(api_key=api_key)
    memory = SummaryMemory(
        client=client,
        model=args.model,
        memory_path=args.memory,
        recent_turns=args.recent_turns,
    )

    print(f"OpenAI Summary Memory Chatbot ({args.model})")
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

        response = client.responses.create(
            model=args.model,
            input=memory.build_input(user_input),
            max_output_tokens=args.max_tokens,
            store=False,
        )
        answer = extract_answer(response)
        print(f"OpenAI: {answer}\n")
        memory.add_turn(user_input, answer)


if __name__ == "__main__":
    main()
