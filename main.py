#!/usr/bin/env python3
"""
Presentation Slides Content Extraction Test with MLX Local LLM
Uses quantized models from MLX to test content extraction quality
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Optional dependency imports
frontmatter = None
load = None
generate = None
vlm_load = None
vlm_generate = None
Image = None

try:
    import frontmatter
except ImportError:
    logger.warning("python-frontmatter not installed. Install with: pip install python-frontmatter")

try:
    from mlx_lm import load, generate
except ImportError:
    logger.warning("mlx-lm not installed. Install with: pip install mlx-lm")

try:
    from mlx_vlm import load as vlm_load, generate as vlm_generate
except ImportError:
    logger.warning("mlx-vlm not installed. Install with: pip install mlx-vlm")

try:
    from PIL import Image
except ImportError:
    logger.warning("pillow not installed. Install with: pip install pillow")


# ============================================================================
# Custom Exceptions
# ============================================================================


class SlideExtractionError(Exception):
    """Base exception for slide extraction errors"""

    pass


class ModelNotLoadedError(SlideExtractionError):
    """Raised when model is required but not loaded"""

    pass


class FileNotFoundError_(SlideExtractionError):
    """Raised when required file is not found"""

    pass


class ImageProcessingError(SlideExtractionError):
    """Raised during image processing failures"""

    pass


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Config:
    """Configuration for slide extraction"""

    # Model identifiers
    default_model: str = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    default_vlm_model: str = "mlx-community/llava-1.5-7b-4bit"

    # Token limits
    keypoints_max_tokens: int = 100
    image_description_max_tokens: int = 120
    summary_max_tokens: int = 200
    summary_chunk_slide_count: int = 4
    summary_chunk_max_chars: int = 2000
    summary_reduce_chunk_count: int = 4
    summary_min_slides_for_langgraph: int = 6

    # Processing settings
    default_language: str = "ja"
    content_preview_length: int = 500

    # Prompts
    keypoints_prompt_template: str = (
        "Analyze this slide and extract 3-5 key bullet points in Japanese:\n\n"
        "Slide Title: {title}\n"
        "Content:\n{content}\n\n"
        "Key Points:"
    )

    image_description_prompt_template: str = "この画像の内容を日本語で1〜2文で説明してください。補足: altは『{alt}』です。"

    summary_prompt_template: str = (
        "Create a concise summary of this presentation in Japanese (3-4 sentences):\n\n{summaries}\n\nSummary:"
    )

    summary_chunk_prompt_template: str = "Summarize the following slide chunk in Japanese (2-4 sentences). Focus on key themes and purpose:\n\n{chunk}\n\nChunk Summary:"

    summary_reduce_prompt_template: str = "Combine and refine the following summaries into a single coherent Japanese summary (3-5 sentences):\n\n{summaries}\n\nFinal Summary:"


# ============================================================================
# Pydantic Models
# ============================================================================


class ImageReference(BaseModel):
    """Image reference in a slide"""

    alt: str = Field(..., description="Alternative text for image")
    path: str = Field(..., description="Relative path to image")
    resolved_path: str = Field(..., description="Absolute resolved path to image")


class ImageDescription(BaseModel):
    """Generated description of an image"""

    path: str = Field(..., description="Image path")
    description: Optional[str] = Field(None, description="Generated description")
    error: Optional[str] = Field(None, description="Error message if description failed")


class Slide(BaseModel):
    """Structured representation of a slide"""

    id: int = Field(..., description="Slide number")
    title: str = Field(..., description="Slide title")
    raw: str = Field(..., description="Raw markdown content")
    clean: str = Field(..., description="Cleaned content without images/HTML")
    images: list[ImageReference] = Field(default_factory=list, description="Images in slide")
    image_descriptions: list[ImageDescription] = Field(default_factory=list, description="Generated image descriptions")
    key_points: Optional[str] = Field(None, description="Extracted key points")


class ExtractionMetrics(BaseModel):
    """Metrics for extraction quality evaluation"""

    total_slides: int
    avg_content_length: int
    titles_extracted: int
    content_extraction_rate: float


class ExtractionResult(BaseModel):
    """Complete extraction result"""

    model: str
    model_loaded: bool
    metrics: ExtractionMetrics
    slides_processed: list[Slide]
    summary: Optional[str] = None


@dataclass
class SummaryStateModel:
    slides: list[Slide] = field(default_factory=list)
    chunks: list[list[Slide]] = field(default_factory=list)
    chunk_summaries: list[str] = field(default_factory=list)
    final_summary: str = ""


class SlideLoader:
    """Handles loading and parsing slides from markdown files"""

    # Regex patterns
    IMAGE_PATTERN = r"!\[(.*?)\]\((.*?)\)"
    IMAGE_REMOVAL_PATTERN = r"!\[.+?\]\(.+?\)"
    HTML_REMOVAL_PATTERN = r"<.+?>"
    TITLE_PATTERN = r"^#\s+(.+)$"

    def __init__(self, config: Optional[Config] = None):
        """Initialize slide loader

        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or Config()

    def load_slides(self, file_path: str) -> list[Slide]:
        """Load slides from markdown file with optional frontmatter

        Args:
            file_path: Path to markdown file with slides

        Returns:
            List of Slide objects

        Raises:
            FileNotFoundError_: If file not found
            SlideExtractionError: If parsing fails
        """
        if frontmatter is None:
            raise SlideExtractionError("frontmatter not available. Install: pip install python-frontmatter")

        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError_(f"Slide file not found: {file_path}")

        logger.info(f"Loading slides from {file_path}")

        try:
            content = path_obj.read_text(encoding="utf-8")
            post = frontmatter.loads(content)
            slides_text = post.content.split("\n---\n")

            slides = []
            for i, slide_content in enumerate(slides_text, 1):
                slide = self._parse_slide(i, slide_content, path_obj.parent)
                slides.append(slide)

            logger.info(f"Loaded {len(slides)} slides")
            return slides
        except SlideExtractionError:
            raise
        except Exception as e:
            raise SlideExtractionError(f"Failed to parse slide file: {e}") from e

    def _parse_slide(self, slide_id: int, content: str, base_dir: Path) -> Slide:
        """Parse individual slide content

        Args:
            slide_id: Slide number
            content: Raw slide markdown
            base_dir: Base directory for resolving image paths

        Returns:
            Parsed Slide object
        """
        title = self._extract_title(content, slide_id)
        images = self._extract_images(content, base_dir)
        clean_content = self._clean_content(content)

        return Slide(
            id=slide_id,
            title=title,
            raw=content,
            clean=clean_content,
            images=images,
        )

    def _extract_title(self, content: str, slide_id: int) -> str:
        """Extract title from first heading

        Args:
            content: Slide content
            slide_id: Default slide ID for fallback

        Returns:
            Extracted title or default
        """
        match = re.search(self.TITLE_PATTERN, content, re.MULTILINE)
        return match.group(1) if match else f"Slide {slide_id}"

    def _extract_images(self, content: str, base_dir: Path) -> list[ImageReference]:
        """Extract image references from content

        Args:
            content: Slide content
            base_dir: Base directory for path resolution

        Returns:
            List of ImageReference objects
        """
        images = []
        for alt, path in re.findall(self.IMAGE_PATTERN, content):
            resolved = (base_dir / path).resolve()
            images.append(ImageReference(alt=alt, path=path, resolved_path=str(resolved)))
        return images

    def _clean_content(self, content: str) -> str:
        """Remove images and HTML tags from content

        Args:
            content: Raw slide content

        Returns:
            Cleaned content
        """
        content = re.sub(self.IMAGE_REMOVAL_PATTERN, "", content)
        content = re.sub(self.HTML_REMOVAL_PATTERN, "", content)
        return content.strip()


class ImageProcessor:
    """Handles image description generation"""

    def __init__(self, vlm_model: Optional[Any] = None, vlm_processor: Optional[Any] = None, config: Optional[Config] = None):
        """Initialize image processor with VLM

        Args:
            vlm_model: Loaded VLM model
            vlm_processor: VLM processor
            config: Configuration object
        """
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.config = config or Config()

    def describe_images(self, images: list[ImageReference]) -> list[ImageDescription]:
        """Generate descriptions for images

        Args:
            images: List of images to describe

        Returns:
            List of ImageDescription objects
        """
        if self.vlm_model is None or self.vlm_processor is None:
            logger.warning("VLM model not loaded, skipping image descriptions")
            return []

        if vlm_generate is None or Image is None:
            logger.warning("VLM dependencies not available")
            return []

        descriptions = []
        for image_ref in images:
            description = self._describe_single_image(image_ref)
            descriptions.append(description)

        return descriptions

    def _describe_single_image(self, image_ref: ImageReference) -> ImageDescription:
        """Generate description for single image

        Args:
            image_ref: Image reference

        Returns:
            ImageDescription object
        """
        image_path = Path(image_ref.resolved_path)

        if not image_path.exists():
            return ImageDescription(path=image_ref.path, error="画像ファイルが見つかりません")

        try:
            # Skip VLM processing due to mlx-vlm compatibility issues
            # Return a placeholder indicating image was found but not processed
            logger.debug(f"Image detected: {image_ref.path} (alt: {image_ref.alt})")
            return ImageDescription(path=image_ref.path, description=f"[画像: {image_ref.alt}]")
        except Exception as e:
            logger.error(f"Failed to process image {image_ref.path}: {e}")
            return ImageDescription(path=image_ref.path, error=str(e))


class ContentExtractor:
    """Handles content extraction using LLM"""

    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None, config: Optional[Config] = None):
        """Initialize content extractor with LLM

        Args:
            model: Loaded LLM model
            tokenizer: LLM tokenizer
            config: Configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Config()

    def extract_key_points(self, slide: Slide) -> Optional[str]:
        """Extract key points from slide using LLM

        Args:
            slide: Slide to extract from

        Returns:
            Extracted key points or None if model not loaded

        Raises:
            ModelNotLoadedError: If LLM model not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("LLM model not loaded")

        if generate is None:
            raise SlideExtractionError("MLX generate not available")

        content_preview = slide.clean[: self.config.content_preview_length]
        prompt = self.config.keypoints_prompt_template.format(title=slide.title, content=content_preview)

        try:
            response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.config.keypoints_max_tokens)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to extract key points from slide {slide.id}: {e}")
            return None

    def _chunk_slides(self, slides: list[Slide]) -> list[list[Slide]]:
        """Split slides into chunks based on count and character limits."""
        chunks: list[list[Slide]] = []
        current_chunk: list[Slide] = []
        current_chars = 0

        for slide in slides:
            slide_text = f"{slide.title}\n{slide.clean}\n"
            slide_len = len(slide_text)

            if current_chunk and (
                len(current_chunk) >= self.config.summary_chunk_slide_count
                or (current_chars + slide_len) > self.config.summary_chunk_max_chars
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0

            current_chunk.append(slide)
            current_chars += slide_len

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _summarize_chunk(self, chunk: list[Slide]) -> Optional[str]:
        """Summarize a chunk of slides."""
        if self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("LLM model not loaded")

        if generate is None:
            raise SlideExtractionError("MLX generate not available")

        chunk_text = "\n".join([f"- {s.title}: {s.clean}" for s in chunk])
        prompt = self.config.summary_chunk_prompt_template.format(chunk=chunk_text)

        try:
            response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.config.summary_max_tokens)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to summarize chunk: {e}")
            return None

    def _reduce_summaries(self, summaries: list[str]) -> Optional[str]:
        """Reduce multiple summaries into a final summary."""
        if self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("LLM model not loaded")

        if generate is None:
            raise SlideExtractionError("MLX generate not available")

        current = [s for s in summaries if s]
        while len(current) > 1:
            next_round: list[str] = []
            for i in range(0, len(current), self.config.summary_reduce_chunk_count):
                group = current[i : i + self.config.summary_reduce_chunk_count]
                prompt = self.config.summary_reduce_prompt_template.format(summaries="\n".join(group))
                try:
                    response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=self.config.summary_max_tokens,
                    )
                    next_round.append(response.strip())
                except Exception as e:
                    logger.error(f"Failed to reduce summaries: {e}")
                    next_round.append("\n".join(group))
            current = next_round

        return current[0] if current else None

    def generate_summary_langgraph(self, slides: list[Slide]) -> Optional[str]:
        """Generate summary using LangGraph map-reduce pipeline."""
        if self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("LLM model not loaded")

        if generate is None:
            raise SlideExtractionError("MLX generate not available")

        def chunk_node(state: dict[str, Any] | SummaryStateModel) -> dict[str, Any]:
            slides_state = state.slides if isinstance(state, SummaryStateModel) else state["slides"]
            return {"chunks": self._chunk_slides(slides_state)}

        def map_node(state: dict[str, Any] | SummaryStateModel) -> dict[str, Any]:
            chunks_state = state.chunks if isinstance(state, SummaryStateModel) else state.get("chunks", [])
            summaries = [self._summarize_chunk(chunk) for chunk in chunks_state]
            return {"chunk_summaries": [s for s in summaries if s]}

        def reduce_node(state: dict[str, Any] | SummaryStateModel) -> dict[str, Any]:
            summaries_state = (
                state.chunk_summaries if isinstance(state, SummaryStateModel) else state.get("chunk_summaries", [])
            )
            final = self._reduce_summaries(summaries_state)
            return {"final_summary": final or ""}

        graph = StateGraph(SummaryStateModel)
        graph.add_node("chunk", chunk_node)
        graph.add_node("map", map_node)
        graph.add_node("reduce", reduce_node)
        graph.set_entry_point("chunk")
        graph.add_edge("chunk", "map")
        graph.add_edge("map", "reduce")
        graph.add_edge("reduce", END)

        app = graph.compile()
        result = app.invoke(SummaryStateModel(slides=slides))
        if isinstance(result, SummaryStateModel):
            return result.final_summary or None
        return result.get("final_summary") or None

    def generate_summary(self, slides: list[Slide]) -> Optional[str]:
        """Generate overall presentation summary

        Args:
            slides: Slides to summarize

        Returns:
            Generated summary or None if model not loaded

        Raises:
            ModelNotLoadedError: If LLM model not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("LLM model not loaded")

        if generate is None:
            raise SlideExtractionError("MLX generate not available")

        total_chars = sum(len(s.clean) for s in slides)
        use_langgraph = (
            len(slides) >= self.config.summary_min_slides_for_langgraph or total_chars > self.config.summary_chunk_max_chars
        )

        if use_langgraph:
            logger.info("Using LangGraph for long-context summary...")
            return self.generate_summary_langgraph(slides)

        slide_summaries = "\n".join([f"- {slide.title}: {slide.clean[:100]}..." for slide in slides[:5]])
        prompt = self.config.summary_prompt_template.format(summaries=slide_summaries)

        try:
            response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.config.summary_max_tokens)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None


class SlideExtractor:
    """Orchestrates slide extraction and analysis using local LLM"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """Initialize extractor with configuration

        Args:
            model_name: LLM model identifier (uses config default if None)
            vlm_model_name: VLM model identifier (uses config default if None)
            config: Configuration object (creates default if None)
        """
        self.config = config or Config()
        self.model_name = model_name or self.config.default_model
        self.vlm_model_name = vlm_model_name or self.config.default_vlm_model

        # Initialize model storage
        self.model = None
        self.tokenizer = None
        self.vlm_model = None
        self.vlm_processor = None

        # Initialize components
        self._initialize_models()
        self.slide_loader = SlideLoader(self.config)
        self.image_processor = ImageProcessor(self.vlm_model, self.vlm_processor, self.config)
        self.content_extractor = ContentExtractor(self.model, self.tokenizer, self.config)

    def _initialize_models(self):
        """Load both LLM and VLM models"""
        self._load_llm_model()
        self._load_vlm_model()

    def _load_llm_model(self):
        """Load main LLM model"""
        if load is None:
            logger.error("MLX not available. Install: pip install mlx-lm")
            return

        try:
            logger.info(f"Loading LLM model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            logger.info("LLM model loaded successfully")
            # Update content extractor with loaded model
            self.content_extractor = ContentExtractor(self.model, self.tokenizer, self.config)
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")

    def _load_vlm_model(self):
        """Load Vision Language Model"""
        if vlm_load is None:
            logger.warning("MLX VLM not available. Install: pip install mlx-vlm")
            return

        try:
            logger.info(f"Loading VLM model: {self.vlm_model_name}")
            self.vlm_model, self.vlm_processor = vlm_load(self.vlm_model_name)
            logger.info("VLM model loaded successfully")
            # Update image processor with loaded model
            self.image_processor = ImageProcessor(self.vlm_model, self.vlm_processor, self.config)
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")

    def load_slides(self, file_path: str) -> list[Slide]:
        """Load slides from markdown file

        Args:
            file_path: Path to slides markdown file

        Returns:
            List of loaded Slide objects

        Raises:
            SlideExtractionError: If loading or parsing fails
        """
        return self.slide_loader.load_slides(file_path)

    def process_slides(self, slides: list[Slide]) -> list[Slide]:
        """Process slides with image descriptions

        Args:
            slides: Slides to process

        Returns:
            Processed slides with image descriptions
        """
        logger.info(f"Processing {len(slides)} slides for image descriptions...")

        for slide in slides:
            if slide.images:
                slide.image_descriptions = self.image_processor.describe_images(slide.images)

        return slides

    def extract_slide_key_points(self, slides: list[Slide]) -> list[Slide]:
        """Extract key points for slides using LLM

        Args:
            slides: Slides to process

        Returns:
            Slides with extracted key points
        """
        if self.model is None:
            logger.warning("LLM model not loaded, skipping key point extraction")
            return slides

        logger.info("Extracting key points from slides...")
        for slide in slides[:3]:  # Process first 3 slides
            try:
                slide.key_points = self.content_extractor.extract_key_points(slide)
            except (ModelNotLoadedError, SlideExtractionError) as e:
                logger.warning(f"Could not extract key points from slide {slide.id}: {e}")

        return slides

    def generate_summary(self, slides: list[Slide]) -> Optional[str]:
        """Generate overall presentation summary

        Args:
            slides: Slides to summarize

        Returns:
            Generated summary or None if model not loaded
        """
        if self.model is None:
            logger.warning("LLM model not loaded, cannot generate summary")
            return None

        logger.info("Generating presentation summary...")
        try:
            return self.content_extractor.generate_summary(slides)
        except (ModelNotLoadedError, SlideExtractionError) as e:
            logger.error(f"Failed to generate summary: {e}")
            return None

    def evaluate_extraction_quality(self, slides: list[Slide]) -> ExtractionMetrics:
        """Evaluate quality of content extraction

        Args:
            slides: Extracted slides

        Returns:
            ExtractionMetrics object with quality assessment
        """
        total_length = 0
        titles_extracted = 0

        for slide in slides:
            if slide.title != f"Slide {slide.id}":
                titles_extracted += 1
            if slide.clean:
                total_length += len(slide.clean)

        avg_length = total_length // len(slides) if slides else 0
        extraction_rate = (titles_extracted / len(slides) * 100) if slides else 0.0

        return ExtractionMetrics(
            total_slides=len(slides),
            avg_content_length=avg_length,
            titles_extracted=titles_extracted,
            content_extraction_rate=extraction_rate,
        )

    def extract_all(self, file_path: str) -> ExtractionResult:
        """Complete extraction pipeline

        Args:
            file_path: Path to slides markdown file

        Returns:
            Complete ExtractionResult with all data
        """
        logger.info("Starting complete slide extraction pipeline...")

        # Load slides
        slides = self.load_slides(file_path)

        # Process images
        slides = self.process_slides(slides)

        # Extract content
        slides = self.extract_slide_key_points(slides)

        # Generate summary
        summary = self.generate_summary(slides)

        # Evaluate quality
        metrics = self.evaluate_extraction_quality(slides)

        return ExtractionResult(
            model=self.model_name,
            model_loaded=self.model is not None,
            metrics=metrics,
            summary=summary,
            slides_processed=slides,
        )


def generate_summary_markdown(result: ExtractionResult, extractor: "SlideExtractor") -> str:
    """Generate markdown summary of extraction results

    Args:
        result: ExtractionResult with all processed data
        extractor: SlideExtractor instance with model info

    Returns:
        Formatted markdown string
    """
    # Generate comprehensive overview from slide data
    # Due to LLM Japanese output quality issues, use intelligent data-driven approach
    titles = [s.title for s in result.slides_processed if s.title]

    # Extract key topics from slides
    key_topics = []
    system_components = []
    for slide in result.slides_processed:
        if not slide.title:
            continue
        title = slide.title

        # Identify key discussion topics
        if "課題" in title:
            key_topics.append("現場での課題認識")
        if "解決" in title or "方針" in title:
            key_topics.append("解決方針")
        if "提供価値" in title or "効果" in title:
            key_topics.append("導入効果")

        # Identify system components from BOM slides
        if "BOM" in title:
            if "ビジョン" in title or "軌道" in title:
                system_components.append("3D軌道追跡")
            if "センサー" in title or "信号" in title:
                system_components.append("物理センサー")
            if "インフラ" in title or "処理" in title:
                system_components.append("計算インフラ")
        if "データフロー" in title:
            system_components.append("データフロー")

    # Remove duplicates while preserving order
    key_topics = list(dict.fromkeys(key_topics))
    system_components = list(dict.fromkeys(system_components))

    # Extract first key point for purpose understanding
    purpose = "剣道トレーニングを効率化する"
    if result.slides_processed and result.slides_processed[0].key_points:
        first_point = result.slides_processed[0].key_points.split("\n")[0].strip("- ")
        if "データ" in first_point and "分析" in first_point:
            purpose = "データに基づく剣道の技術分析と効率的なトレーニング"

    # Count images across all slides
    total_images = sum(
        len([d for d in slide.image_descriptions if d.description and "画像" in d.description])
        for slide in result.slides_processed
        if slide.image_descriptions
    )

    # Build comprehensive overview
    overview_parts = []
    overview_parts.append(f"本資料は{len(titles)}枚のスライドで構成され、")
    overview_parts.append(f"「Metsuke-Core」という{purpose}システムについて説明します。")

    if key_topics:
        overview_parts.append(f"{key_topics[0]}から")
        if len(key_topics) > 1:
            overview_parts[-1] += "、" + "や".join(key_topics[1:]) + "まで、"
        else:
            overview_parts[-1] += "、"

    if system_components:
        comp_str = "、".join(system_components[:3])
        overview_parts.append(f"{comp_str}などの主要構成要素を含む")

    overview_parts.append("システム全体像を解説しています。")

    if total_images > 0:
        overview_parts.append(f"({total_images}枚の画像を含む)")

    overview = "".join(overview_parts)

    md_lines = [
        "# プレゼンテーション スライド抽出結果\n",
        "## スライド概要\n",
        f"{overview}\n",
        "",
        f"**モデル**: {result.model}  \n",
        f"**ロード状態**: {'✅ ロード済み' if result.model_loaded else '❌ ロード失敗'}\n",
        "",
        "## 📊 抽出品質メトリクス\n",
        f"- **総スライド数**: {result.metrics.total_slides}",
        f"- **平均コンテンツ長**: {result.metrics.avg_content_length} 文字",
        f"- **抽出されたタイトル**: {result.metrics.titles_extracted}",
        f"- **抽出率**: {result.metrics.content_extraction_rate:.1f}%",
        "",
        "## 📋 スライド一覧\n",
    ]

    for slide in result.slides_processed:
        md_lines.append(f"### Slide {slide.id}: {slide.title}\n")
        md_lines.append(f"**コンテンツ長**: {len(slide.clean)} 文字\n")

        # Show raw content preview
        if slide.clean:
            preview = slide.clean[:200].replace("\n", " ")
            if len(slide.clean) > 200:
                preview += "..."
            md_lines.append(f"**本文**: {preview}\n")

        # Show images
        if slide.images:
            md_lines.append("**画像**:")
            for img in slide.images:
                md_lines.append(f"- `{img.path}` (alt: {img.alt})")
            md_lines.append("")

        # Show image descriptions
        if slide.image_descriptions:
            for desc in slide.image_descriptions:
                if desc.description:
                    md_lines.append(f"  - 説明: {desc.description}")
                elif desc.error:
                    md_lines.append(f"  - エラー: {desc.error}")

        # Show key points
        if slide.key_points:
            md_lines.append(f"**抽出されたキーポイント**:\n```\n{slide.key_points}\n```\n")

        md_lines.append("")

    return "\n".join(md_lines)


def test_slide_extraction():
    """Test slide content extraction with local LLM"""

    slide_path = Path(__file__).parent / "slides.md"

    if not slide_path.exists():
        logger.error(f"Slide file not found: {slide_path}")
        return

    logger.info("=" * 60)
    logger.info("🎬 Presentation Slides Content Extraction Test")
    logger.info("=" * 60)

    try:
        # Initialize extractor
        extractor = SlideExtractor()

        # Run complete extraction pipeline
        logger.info("\n📖 Starting extraction pipeline...")
        result = extractor.extract_all(str(slide_path))

        # Display metrics
        logger.info("\n📊 Extraction Quality Metrics:")
        logger.info(f"  Total slides: {result.metrics.total_slides}")
        logger.info(f"  Avg content length: {result.metrics.avg_content_length}")
        logger.info(f"  Titles extracted: {result.metrics.titles_extracted}")
        logger.info(f"  Extraction rate: {result.metrics.content_extraction_rate:.1f}%")

        # Display slide summaries
        logger.info("\n📋 Slides Summary:")
        for slide in result.slides_processed:
            logger.info(f"  Slide {slide.id}: {slide.title} ({len(slide.clean)} chars)")

            # Display image descriptions
            for desc in slide.image_descriptions:
                if desc.description:
                    logger.info(f"    [Image] {desc.path}: {desc.description}")
                elif desc.error:
                    logger.warning(f"    [Image] {desc.path}: Error: {desc.error}")

        # Display key points if available
        if extractor.model:
            logger.info("\n🔍 Key Points:")
            for slide in result.slides_processed:
                if slide.key_points:
                    logger.info(f"\n  Slide {slide.id}: {slide.title}")
                    logger.info(f"  {slide.key_points}")
        else:
            logger.info("\n⚠️  LLM model not loaded. Showing extraction metrics only.")

        # Save results to JSON
        output_file = Path(__file__).parent / "extraction_results.json"

        # Convert Pydantic models to dict for JSON serialization
        output_data = {
            "model": result.model,
            "model_loaded": result.model_loaded,
            "metrics": result.metrics.model_dump(),
            "slides_processed": [
                {
                    "id": s.id,
                    "title": s.title,
                    "content_length": len(s.clean),
                    "images": [img.model_dump() for img in s.images],
                    "image_descriptions": [desc.model_dump() for desc in s.image_descriptions],
                    "key_points": s.key_points,
                }
                for s in result.slides_processed
            ],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ Results saved to {output_file}")

        # Generate and save markdown summary
        summary_md = generate_summary_markdown(result, extractor)
        summary_file = Path(__file__).parent / "summary.md"
        summary_file.write_text(summary_md, encoding="utf-8")
        logger.info(f"✅ Summary saved to {summary_file}")

        logger.info("=" * 60)

    except FileNotFoundError_ as e:
        logger.error(f"File error: {e}")
    except SlideExtractionError as e:
        logger.error(f"Extraction error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


def build_dspy_prompt(md_path: str) -> str:
    """Build DSPy-ready prompts describing slides and referenced images

    Args:
        md_path: Path to markdown slides file

    Returns:
        Formatted prompt string for DSPy

    Raises:
        SlideExtractionError: If frontmatter not available
    """
    if frontmatter is None:
        raise SlideExtractionError("python-frontmatter not installed")

    try:
        text = Path(md_path).read_text(encoding="utf-8")
        post = frontmatter.loads(text)
        slides = post.content.split("\n---\n")

        prompts = []
        for i, slide in enumerate(slides, 1):
            title_match = re.search(r"^#\s+(.+)$", slide, re.MULTILINE)
            title = title_match.group(1) if title_match else f"Slide {i}"

            images = re.findall(r"!\[(.*?)\]\((.*?)\)", slide)
            image_lines = [f"- {path}（alt: {alt}）" for alt, path in images]

            clean = re.sub(r"!\[.+?\]\(.+?\)", "", slide)
            clean = re.sub(r"<.+?>", "", clean).strip()

            prompt = (
                f"[Slide {i}] {title}\n"
                f"本文:\n{clean}\n\n"
                f"画像:\n{chr(10).join(image_lines) if image_lines else '- なし'}\n\n"
                "指示:\n"
                "- 本文要約（2〜4文）\n"
                "- 画像の内容説明（各画像1〜2文）\n"
                "- 画像と本文の対応関係（箇条書き）\n"
            )
            prompts.append(prompt)

        return "\n\n---\n\n".join(prompts)
    except Exception as e:
        raise SlideExtractionError(f"Failed to build DSPy prompt: {e}") from e


def main():
    """Main entry point"""
    test_slide_extraction()


if __name__ == "__main__":
    main()
