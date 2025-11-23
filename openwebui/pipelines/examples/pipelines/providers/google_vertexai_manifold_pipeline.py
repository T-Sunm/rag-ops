"""
title: Google GenAI (Vertex AI) Manifold Pipeline
author: Hiromasa Kakehashi & Olv Grolle
date: 2024-09-19
version: 1.0
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: vertexai
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
usage_instructions:
  To use Gemini with the Vertex AI API, a service account with the appropriate role (e.g., `roles/aiplatform.user`) is required.
  - For deployment on Google Cloud: Associate the service account with the deployment.
  - For use outside of Google Cloud: Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the service account key file.
"""

import os
import base64
from typing import Iterator, List, Union

import vertexai
from pydantic import BaseModel, Field
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)


class Pipeline:
    """Google GenAI pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_PROJECT_ID: str = ""
        GOOGLE_CLOUD_REGION: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.name = "VertexAI: "

        self.valves = self.Valves(
            **{
                "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID", ""),
                "GOOGLE_CLOUD_REGION": os.getenv("GOOGLE_CLOUD_REGION", ""),
                "USE_PERMISSIVE_SAFETY": False,
            }
        )
        self.pipelines = [

            # Gemini 2.0 models
            {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash-Lite"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
            # Gemini 2.5 models
            {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash-Lite"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro "},

             ]

    async def on_startup(self) -> None:
        """This function is called when the server is started."""

        print(f"on_startup:{__name__}")
        vertexai.init(
            project=self.valves.GOOGLE_PROJECT_ID,
            location=self.valves.GOOGLE_CLOUD_REGION,
        )

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        vertexai.init(
            project=self.valves.GOOGLE_PROJECT_ID,
            location=self.valves.GOOGLE_CLOUD_REGION,
        )

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Iterator]:
        try:
            if not (model_id.startswith("gemini-") or model_id.startswith("gemma-")):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")
            print(f"Received {len(messages)} messages from OpenWebUI")
            
            # Debug: Log message structure
            for i, msg in enumerate(messages):
                print(f"Message {i}: role={msg.get('role')}, content type={type(msg.get('content'))}")
                if isinstance(msg.get('content'), list):
                    for j, content_part in enumerate(msg['content']):
                        print(f"  Part {j}: type={content_part.get('type')}")
                        if content_part.get('type') == 'image_url':
                            img_url = content_part.get('image_url', {}).get('url', '')
                            print(f"    Image URL prefix: {img_url[:50]}...")

            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            model = GenerativeModel(
                model_name=model_id,
                system_instruction=system_message,
            )

            if body.get("title", False):  # If chat title generation is requested
                contents = [Content(role="user", parts=[Part.from_text(user_message)])]
                print("Title generation mode - using simple text content")
            else:
                contents = self.build_conversation_history(messages)

            # Log what we're sending to Vertex AI
            print(f"Sending {len(contents)} messages to Vertex AI:")
            for i, content in enumerate(contents):
                print(f"  Message {i}: role={content.role}, parts={len(content.parts)}")
                for j, part in enumerate(content.parts):
                    if hasattr(part, '_raw_data') and part._raw_data:
                        print(f"    Part {j}: Image data ({len(part._raw_data)} bytes)")
                    else:
                        part_text = str(part)[:100] if str(part) else "No text"
                        print(f"    Part {j}: Text - {part_text}...")

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            print("Calling Vertex AI generate_content...")
            response = model.generate_content(
                contents,
                stream=body.get("stream", False),
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.text

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                print(f"Chunk: {chunk.text}")
                yield chunk.text

    def build_conversation_history(self, messages: List[dict]) -> List[Content]:
        contents = []

        for message in messages:
            if message["role"] == "system":
                continue

            parts = []

            if isinstance(message.get("content"), list):
                print(f"Processing multi-part message with {len(message['content'])} parts")
                for content in message["content"]:
                    print(f"Processing content type: {content.get('type', 'unknown')}")
                    if content["type"] == "text":
                        parts.append(Part.from_text(content["text"]))
                        print(f"Added text part: {content['text'][:50]}...")
                    elif content["type"] == "image_url":
                        image_url = content["image_url"]["url"]
                        print(f"Processing image URL (first 50 chars): {image_url[:50]}...")
                        if image_url.startswith("data:image"):
                            try:
                                # Split the data URL to get mime type and base64 data
                                header, image_data = image_url.split(',', 1)
                                mime_type = header.split(':')[1].split(';')[0]
                                print(f"Detected image MIME type: {mime_type}")
                                
                                # Validate supported image formats
                                supported_formats = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
                                if mime_type not in supported_formats:
                                    print(f"ERROR: Unsupported image format: {mime_type}")
                                    continue
                                
                                # Decode the base64 image data
                                decoded_image_data = base64.b64decode(image_data)
                                print(f"Successfully decoded image data: {len(decoded_image_data)} bytes")
                                
                                # Create the Part object with the image data 
                                image_part = Part.from_data(decoded_image_data, mime_type=mime_type)
                                parts.append(image_part)
                                print(f"Successfully added image part to conversation")
                            except Exception as e:
                                print(f"ERROR processing image: {e}")
                                import traceback
                                traceback.print_exc()
                                continue
                        else:
                            # Handle image URLs
                            print(f"Processing external image URL: {image_url}")
                            parts.append(Part.from_uri(image_url))
            else:
                parts = [Part.from_text(message["content"])]
                print(f"Added simple text message: {message['content'][:50]}...")

            role = "user" if message["role"] == "user" else "model"
            print(f"Creating Content with role='{role}' and {len(parts)} parts")
            contents.append(Content(role=role, parts=parts))

        print(f"Built conversation history with {len(contents)} messages")
        return contents
