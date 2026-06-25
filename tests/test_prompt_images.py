"""Tests for multimodal message parsing on the HTTP server path."""

import base64
import unittest

from server.prompt import has_image_placeholder_without_bytes, turn_image, turn_prompt
from server.schemas import ChatMessage


def _msg(role: str, content) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TurnImageTests(unittest.TestCase):
    def test_extracts_openai_base64_image_url(self):
        raw = b"\x89PNG\r\n\x1a\nfake"
        b64 = base64.b64encode(raw).decode("ascii")
        messages = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "Reverse-engineer the prompt for this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            )
        ]
        self.assertEqual(turn_image(messages), raw)

    def test_ignores_messages_without_images(self):
        messages = [_msg("user", "Hello")]
        self.assertIsNone(turn_image(messages))

    def test_uses_only_latest_user_message(self):
        old = base64.b64encode(b"old").decode("ascii")
        new = base64.b64encode(b"new").decode("ascii")
        messages = [
            _msg("user", [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{old}"}}]),
            _msg("assistant", "done"),
            _msg(
                "user",
                [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{new}"}}],
            ),
        ]
        self.assertEqual(turn_image(messages), b"new")

    def test_invalid_base64_raises(self):
        messages = [
            _msg("user", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}}]),
        ]
        with self.assertRaises(ValueError):
            turn_image(messages)

    def test_multiple_images_in_one_turn_uses_first_only(self):
        first = base64.b64encode(b"first-image").decode("ascii")
        second = base64.b64encode(b"second-image").decode("ascii")
        messages = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "Describe the first image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{first}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{second}"}},
                ],
            )
        ]
        self.assertEqual(turn_image(messages), b"first-image")

    def test_file_part_with_base64_data(self):
        raw = b"\x89PNG\r\n\x1a\nfile-part"
        b64 = base64.b64encode(raw).decode("ascii")
        messages = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "Describe this."},
                    {"type": "file", "mediaType": "image/png", "data": b64},
                ],
            )
        ]
        self.assertEqual(turn_image(messages), raw)

    def test_cherry_studio_image_part_with_raw_base64(self):
        raw = b"\x89PNG\r\n\x1a\ncherry"
        b64 = base64.b64encode(raw).decode("ascii")
        messages = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "Reverse-engineer the prompt."},
                    {"type": "image", "image": b64, "mediaType": "image/png"},
                ],
            )
        ]
        self.assertEqual(turn_image(messages), raw)

    def test_split_user_bubbles_image_then_text(self):
        raw = b"\xff\xd8\xffsplit"
        b64 = base64.b64encode(raw).decode("ascii")
        messages = [
            _msg("user", [{"type": "image", "image": b64, "mediaType": "image/jpeg"}]),
            _msg("user", "What is in the image?"),
        ]
        self.assertEqual(turn_image(messages), raw)


class TurnPromptImageTests(unittest.TestCase):
    def test_default_prompt_when_image_only(self):
        raw = base64.b64encode(b"\xff\xd8\xff").decode("ascii")
        messages = [
            _msg("user", [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{raw}"}}]),
        ]
        self.assertEqual(turn_prompt(messages, image_attached=True), "Describe this image in detail.")

    def test_keeps_user_text_with_image(self):
        messages = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "Write a Stable Diffusion prompt."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}} ,
                ],
            )
        ]
        self.assertEqual(turn_prompt(messages, image_attached=True), "Write a Stable Diffusion prompt.")


class ImagePlaceholderTests(unittest.TestCase):
    def test_detects_placeholder_without_bytes(self):
        messages = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "text", "text": "[Image: image/png]"},
                ],
            )
        ]
        self.assertTrue(has_image_placeholder_without_bytes(messages))

    def test_image_part_without_payload_is_detected(self):
        messages = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "Describe this."},
                    {"type": "file", "file": {"filename": "shot.png"}},
                ],
            )
        ]
        self.assertTrue(has_image_placeholder_without_bytes(messages))


if __name__ == "__main__":
    unittest.main()
