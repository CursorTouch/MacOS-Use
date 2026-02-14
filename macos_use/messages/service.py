from pydantic import BaseModel,ConfigDict
from textwrap import shorten
from typing import Literal
from PIL.Image import Image
from io import BytesIO
import base64

class BaseMessage(BaseModel):
    role: Literal["system", "human", "ai", "tool"]
    content: str | None = None
    thinking: str | None = None
    thinking_signature: str | bytes | None = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"
    content: str

    def __repr__(self) -> str:
        return f"SystemMessage(content={shorten(self.content, width=80,placeholder='...')})"

class HumanMessage(BaseMessage):
    role: Literal["human"] = "human"
    content: str

    def __repr__(self) -> str:
        return f"HumanMessage(content={shorten(self.content, width=80,placeholder='...')})"
    
class ImageMessage(BaseMessage):
    role: Literal["human"] = "human"
    content: str
    image: Image|None = None
    images: list[Image] = []
    mime_type: str="image/png"

    # Maximum raw image size in bytes (API limit is 5 MB; use 4.8 MB for safety)
    _MAX_IMAGE_BYTES: int = 4_800_000
    
    @staticmethod
    def _compress_image(img: Image, mime_type: str, max_bytes: int = 4_800_000) -> tuple[bytes, str]:
        """Compress an image to fit within *max_bytes*.

        Returns:
            A tuple of (image_bytes, actual_mime_type).

        Strategy:
        1. Try saving in the requested format with quality=85.
        2. If still too large, fall back to JPEG and reduce quality progressively.
        3. If still too large, resize the image (halve dimensions) and repeat.
        """
        def _save(image: Image, fmt: str, quality: int) -> bytes:
            buf = BytesIO()
            # Ensure RGB mode for JPEG (JPEG doesn't support alpha)
            save_img = image.convert("RGB") if fmt.upper() == "JPEG" else image
            save_img.save(buf, format=fmt, quality=quality)
            return buf.getvalue()

        img_format = mime_type.split("/")[-1].upper()
        if img_format == "JPG":
            img_format = "JPEG"
        original_mime = mime_type

        # First attempt: original format, quality 85
        data = _save(img, img_format, 85)
        if len(data) <= max_bytes:
            return data, original_mime

        # Switch to JPEG for much better compression
        img_format = "JPEG"
        actual_mime = "image/jpeg"

        for quality in (80, 60, 40, 25):
            data = _save(img, img_format, quality)
            if len(data) <= max_bytes:
                return data, actual_mime

        # Still too large – progressively resize
        current = img
        for _ in range(5):
            new_w = max(current.width // 2, 320)
            new_h = max(current.height // 2, 240)
            current = current.resize((new_w, new_h), resample=3)  # LANCZOS
            data = _save(current, img_format, 50)
            if len(data) <= max_bytes:
                return data, actual_mime

        # Last resort: return whatever we have
        return data, actual_mime

    def image_to_base64(self) -> str:
        image_bytes = self.image_to_bytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return base64_image
    
    def scale_image(self, scale: float=0.5) -> None:
        if self.image:
            size=(int(self.image.width * scale), int(self.image.height * scale))
            self.image = self.image.resize(size=size)

    def scale_images(self, scale: float=0.5) -> None:
        if self.image:
            self.scale_image(scale)
        for i in range(len(self.images)):
            size=(int(self.images[i].width * scale), int(self.images[i].height * scale))
            self.images[i] = self.images[i].resize(size=size)

    def image_to_bytes(self) -> bytes:
        data, actual_mime = self._compress_image(self.image, self.mime_type, self._MAX_IMAGE_BYTES)
        self.mime_type = actual_mime
        return data

    def convert_images(self, format: str="base64") -> list[str|bytes]:
        """Convert all images to base64 strings or raw bytes.

        Side effect: updates ``self.mime_type`` to reflect the actual format
        used after compression (e.g. ``image/jpeg`` when the original PNG was
        too large).  LLM providers that read ``msg.mime_type`` after calling
        this method will therefore get the correct value.
        """
        results = []
        actual_mime = self.mime_type
        target_images = self.images if self.images else ([self.image] if self.image else [])
        for img in target_images:
            data, actual_mime = self._compress_image(img, self.mime_type, self._MAX_IMAGE_BYTES)
            if format == "base64":
                results.append(base64.b64encode(data).decode("utf-8"))
            else:
                results.append(data)
        # Update mime_type so downstream consumers (LLM providers) use the
        # correct media type header.
        self.mime_type = actual_mime
        return results

    def __repr__(self) -> str:
        img_desc = shorten(str(self.image), width=30) if self.image else f"{len(self.images)} images"
        return f"ImageMessage(content={shorten(self.content, width=80,placeholder='...')}, image={img_desc}, mime_type={self.mime_type})"

class AIMessage(BaseMessage):
    role: Literal["ai"] = "ai"
    content: str | None = None

    def __repr__(self) -> str:
        return f"AIMessage(content={self.content}, thinking={shorten(str(self.thinking), width=50, placeholder='...')})"

class ToolMessage(BaseMessage):
    role: Literal["tool"] = "tool"
    id: str  # Tool call id
    name: str # Tool name
    params: dict = {} # Tool parameters
    content: str | None = None # Tool result

    def __repr__(self) -> str:
        return f"ToolMessage(name={self.name}, id={self.id}, params={self.params}, content={shorten(self.content, width=80,placeholder='...')}, thinking={shorten(str(self.thinking), width=50, placeholder='...')})"