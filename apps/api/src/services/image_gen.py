import os
import base64
from typing import Optional
from google import genai
from google.genai import types

async def generate_image(prompt: str, is_character: bool = False) -> Optional[str]:
    """
    Calls the Gemini 3.1 Flash Image API to generate an image.
    Returns a base64 data URI string if successful, else None.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: API key not set for image generation.")
        return None

    # Default to 16:9 for backgrounds, 1:1 for heroes
    aspect_ratio = "1:1" if is_character else "16:9"
    
    # Prepend style instructions
    if not is_character:
        full_prompt = f"A beautiful, immersive, slightly dreamy, kid-friendly 3D video game background scene: {prompt}"
    else:
        full_prompt = f"A lively character portrait: {prompt}. Clean background, vibrant colors."

    try:
        client = genai.Client(api_key=api_key)
        
        result = await client.aio.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=full_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio=aspect_ratio
            )
        )
        
        if result.generated_images and len(result.generated_images) > 0:
            image_bytes = result.generated_images[0].image.image_bytes
            b64_data = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_data}"
            
        return None
    except Exception as e:
        print(f"Error generating image with Gemini: {e}")
        return None
