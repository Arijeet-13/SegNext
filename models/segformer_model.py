import logging
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

logger = logging.getLogger(__name__)

def load_segformer_b3(
    num_classes: int,
    checkpoint: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    image_size: int = 768,
):
    """
    Load a SegFormer model and its corresponding image processor.
    """
    print(f"Loading Segformer model from checkpoint: {checkpoint}")

    # Load Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # Load Processor
    try:
        processor = SegformerImageProcessor.from_pretrained(
            checkpoint,
            size={"height": image_size, "width": image_size} if isinstance(image_size, int) else image_size,
        )
    except Exception as e:
        print(f"Processor config not found in {checkpoint}, falling back to default.")
        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            size={"height": image_size, "width": image_size} if isinstance(image_size, int) else image_size,
        )

    return model, processor
