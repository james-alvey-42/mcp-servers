#!/usr/bin/env python3
"""
Modern Minimalist GUI Chat Application for LLM API Bridge

A beautiful Tkinter-based GUI with modern design principles:
- Clean light theme
- Subtle rounded corners
- Real-time status updates
- Responsive layout
"""

import os
import asyncio
import base64
import mimetypes
import uuid
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from typing import List, Dict
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk

# Import our provider system
from providers import (
    LLMProvider,
    LLMMessage,
    ContentPart,
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider,
)


class ModernChatGUI:
    # Modern light theme color palette
    COLORS = {
        "bg_primary": "#FFFFFF",  # Main background
        "bg_secondary": "#F7F9FC",  # Card background
        "bg_tertiary": "#F0F4F8",  # Input background
        "border": "#E2E8F0",  # Borders
        "border_focus": "#94A3B8",  # Focused borders
        "text_primary": "#1E293B",  # Main text
        "text_secondary": "#64748B",  # Secondary text
        "text_dim": "#94A3B8",  # Dimmed text
        "accent": "#3B82F6",  # Primary accent (blue)
        "accent_hover": "#2563EB",  # Accent hover
        "user": "#3B82F6",  # User messages
        "assistant": "#10B981",  # Assistant messages
        "system": "#64748B",  # System messages
        "error": "#EF4444",  # Error messages
        "success": "#10B981",  # Success indicators
        "warning": "#F59E0B",  # Warning messages
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Platypus Chat")
        self.root.geometry("1100x800")

        # Set light theme
        self.root.configure(bg=self.COLORS["bg_primary"])

        # Load logo
        self.logo_image = None
        self.load_logo()

        # Conversation history
        self.conversation_history: List[LLMMessage] = []

        # Provider instances cache
        self.providers: Dict[str, LLMProvider] = {}

        # Status tracking
        self.is_waiting = False

        # Attached files for multimodal messages
        self.attached_files: List[str] = []

        # Content parts for building the message (mix of files and text markers)
        self.content_parts: List[Dict] = []  # List of {"type": "file"/"text", "value": path/text, "id": unique_id}

        # Available models per provider
        self.provider_models = {
            "openai": [
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
                "o3",
                "o3-mini",
                "gpt-4o",
                "gpt-4",
            ],
            "gemini": [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash-exp",
            ],
            "anthropic": [
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-haiku-4-5",
                "claude-sonnet-4-0",
                "claude-opus-4-0",
                "claude-3-7-sonnet-latest",
                "claude-3-5-haiku-latest",
            ],
        }

        # Model-specific requirements
        self.model_requirements = {
            "gpt-5": {"temperature": 1.0, "reason": "GPT-5 requires temperature=1.0"},
            "o3": {
                "temperature": 1.0,
                "reason": "O3 reasoning model requires temperature=1.0",
            },
            "o3-mini": {
                "temperature": 1.0,
                "reason": "O3-mini reasoning model requires temperature=1.0",
            },
        }

        # Create UI
        self.create_widgets()

        # Initialize providers
        self.initialize_providers()

    def load_logo(self):
        """Load and prepare the platypus logo"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(script_dir, "platypus.png")

            # Load and resize the image
            image = Image.open(logo_path)
            # Resize to a reasonable size (40x40 pixels for header)
            image = image.resize((40, 40), Image.Resampling.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(image)
        except Exception as e:
            print(f"Warning: Could not load logo: {e}")
            self.logo_image = None

    def create_widgets(self):
        """Create all GUI widgets with modern design"""

        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.COLORS["bg_primary"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)

        # === HEADER SECTION ===
        header_frame = tk.Frame(main_container, bg=self.COLORS["bg_primary"])
        header_frame.pack(fill=tk.X, pady=(0, 20))

        # Logo and title container
        title_container = tk.Frame(header_frame, bg=self.COLORS["bg_primary"])
        title_container.pack(side=tk.LEFT)

        # Logo (if available)
        if self.logo_image:
            logo_label = tk.Label(
                title_container,
                image=self.logo_image,
                bg=self.COLORS["bg_primary"],
            )
            logo_label.pack(side=tk.LEFT, padx=(0, 12))

        # Title
        title_label = tk.Label(
            title_container,
            text="Platypus Chat",
            font=("Helvetica Neue", 26, "bold"),
            bg=self.COLORS["bg_primary"],
            fg=self.COLORS["text_primary"],
        )
        title_label.pack(side=tk.LEFT)

        # Status indicator (right side)
        self.status_label = tk.Label(
            header_frame,
            text="Ready",
            font=("Helvetica Neue", 12),
            bg=self.COLORS["bg_primary"],
            fg=self.COLORS["text_secondary"],
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # === CONTROLS SECTION ===
        controls_frame = tk.Frame(
            main_container,
            bg=self.COLORS["bg_secondary"],
            highlightbackground=self.COLORS["border"],
            highlightthickness=1,
        )
        controls_frame.pack(fill=tk.X, pady=(0, 15))

        # Inner padding frame
        controls_inner = tk.Frame(controls_frame, bg=self.COLORS["bg_secondary"])
        controls_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Provider selection
        tk.Label(
            controls_inner,
            text="Provider",
            font=("Helvetica Neue", 11),
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["text_secondary"],
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))

        self.provider_var = tk.StringVar(value="gemini")
        self.provider_combo = ttk.Combobox(
            controls_inner,
            textvariable=self.provider_var,
            values=["openai", "gemini", "anthropic"],
            state="readonly",
            width=12,
            font=("Helvetica Neue", 11),
        )
        self.provider_combo.grid(row=0, column=1, padx=(0, 20))
        self.provider_combo.bind("<<ComboboxSelected>>", self.on_provider_changed)

        # Model selection
        tk.Label(
            controls_inner,
            text="Model",
            font=("Helvetica Neue", 11),
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["text_secondary"],
        ).grid(row=0, column=2, sticky=tk.W, padx=(0, 8))

        self.model_var = tk.StringVar(value="gemini-2.5-pro")
        self.model_combo = ttk.Combobox(
            controls_inner,
            textvariable=self.model_var,
            values=self.provider_models["gemini"],
            state="readonly",
            width=25,
            font=("Helvetica Neue", 11),
        )
        self.model_combo.grid(row=0, column=3, padx=(0, 20))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)

        # Temperature
        self.temp_label = tk.Label(
            controls_inner,
            text="Temperature",
            font=("Helvetica Neue", 11),
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["text_secondary"],
        )
        self.temp_label.grid(row=0, column=4, sticky=tk.W, padx=(0, 8))

        self.temp_var = tk.DoubleVar(value=0.7)
        temp_spin = ttk.Spinbox(
            controls_inner,
            from_=0.0,
            to=1.0,
            increment=0.1,
            textvariable=self.temp_var,
            width=5,
            font=("Helvetica Neue", 11),
        )
        temp_spin.grid(row=0, column=5, padx=(0, 30))

        # Clear button - simple text button
        clear_btn = tk.Button(
            controls_inner,
            text="Clear",
            command=self.clear_conversation,
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["text_secondary"],
            font=("Helvetica Neue", 11),
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=8,
        )
        clear_btn.grid(row=0, column=6)

        # Hover effects for clear button
        def on_clear_enter(e):
            clear_btn.config(fg=self.COLORS["text_primary"])

        def on_clear_leave(e):
            clear_btn.config(fg=self.COLORS["text_secondary"])

        clear_btn.bind("<Enter>", on_clear_enter)
        clear_btn.bind("<Leave>", on_clear_leave)

        # === CHAT DISPLAY SECTION ===
        chat_container = tk.Frame(
            main_container,
            bg=self.COLORS["bg_secondary"],
            highlightbackground=self.COLORS["border"],
            highlightthickness=1,
        )
        chat_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Chat display with custom styling
        self.chat_display = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            font=("Menlo", 11),
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["text_primary"],
            insertbackground=self.COLORS["text_primary"],
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=20,
            pady=15,
            spacing3=6,
            state=tk.DISABLED,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure text tags
        self.chat_display.tag_config(
            "user",
            foreground=self.COLORS["user"],
            font=("Helvetica Neue", 11, "bold"),
            spacing1=12,
        )
        self.chat_display.tag_config(
            "assistant",
            foreground=self.COLORS["assistant"],
            font=("Helvetica Neue", 11, "bold"),
            spacing1=12,
        )
        self.chat_display.tag_config(
            "user_message",
            foreground=self.COLORS["text_primary"],
            font=("Menlo", 11),
            lmargin1=20,
            lmargin2=20,
        )
        self.chat_display.tag_config(
            "assistant_message",
            foreground=self.COLORS["text_primary"],
            font=("Menlo", 11),
            lmargin1=20,
            lmargin2=20,
        )
        self.chat_display.tag_config(
            "system",
            foreground=self.COLORS["text_dim"],
            font=("Helvetica Neue", 10),
            spacing1=3,
            spacing3=3,
        )
        self.chat_display.tag_config(
            "error",
            foreground=self.COLORS["error"],
            font=("Helvetica Neue", 10, "bold"),
            spacing1=3,
        )

        # === INPUT SECTION ===
        input_container = tk.Frame(
            main_container,
            bg=self.COLORS["bg_secondary"],
            highlightbackground=self.COLORS["border"],
            highlightthickness=1,
        )
        input_container.pack(fill=tk.X)

        # Input text area with file attachment support
        input_inner = tk.Frame(input_container, bg=self.COLORS["bg_secondary"])
        input_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left side: Input and content preview
        input_left = tk.Frame(input_inner, bg=self.COLORS["bg_secondary"])
        input_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Content parts preview area (shows the order of attachments and text)
        self.content_preview_frame = tk.Frame(
            input_left,
            bg=self.COLORS["bg_tertiary"],
            highlightbackground=self.COLORS["border"],
            highlightthickness=1,
        )
        # Initially hidden, will be shown when content parts exist

        self.input_text = scrolledtext.ScrolledText(
            input_left,
            wrap=tk.WORD,
            font=("Menlo", 11),
            bg=self.COLORS["bg_primary"],
            fg=self.COLORS["text_primary"],
            insertbackground=self.COLORS["accent"],
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
            highlightcolor=self.COLORS["accent"],
            relief="flat",
            padx=12,
            pady=10,
            height=3,
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        self.input_text.bind("<Control-Return>", lambda e: self.send_message())
        self.input_text.bind("<KeyRelease>", self.on_text_changed)

        # Enable drag-and-drop on input text area
        self.input_text.drop_target_register(DND_FILES)
        self.input_text.dnd_bind("<<Drop>>", self.on_file_drop)

        # Right side: Buttons
        buttons_frame = tk.Frame(input_inner, bg=self.COLORS["bg_secondary"])
        buttons_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Attach file button
        self.attach_btn = tk.Button(
            buttons_frame,
            text="📎",
            command=self.attach_file,
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["text_secondary"],
            font=("Helvetica Neue", 16),
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
        )
        self.attach_btn.pack(side=tk.TOP, pady=(0, 5))

        # Attach button hover effects
        def on_attach_enter(e):
            if not self.is_waiting:
                self.attach_btn.config(fg=self.COLORS["text_primary"])

        def on_attach_leave(e):
            if not self.is_waiting:
                self.attach_btn.config(fg=self.COLORS["text_secondary"])

        self.attach_btn.bind("<Enter>", on_attach_enter)
        self.attach_btn.bind("<Leave>", on_attach_leave)

        # Send button - simple, clean design
        self.send_btn = tk.Button(
            buttons_frame,
            text="Send",
            command=self.send_message,
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["text_primary"],
            font=("Helvetica Neue", 12),
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            padx=25,
            pady=10,
        )
        self.send_btn.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Send button hover effects
        def on_send_enter(e):
            if not self.is_waiting:
                self.send_btn.config(bg=self.COLORS["border"], relief="solid")

        def on_send_leave(e):
            if not self.is_waiting:
                self.send_btn.config(bg=self.COLORS["bg_secondary"], relief="solid")

        self.send_btn.bind("<Enter>", on_send_enter)
        self.send_btn.bind("<Leave>", on_send_leave)

    def update_status(self, status_text: str, color: str = None):
        """Update the status indicator"""
        if color is None:
            color = self.COLORS["text_secondary"]
        self.status_label.config(text=status_text, fg=color)
        self.root.update()

    def initialize_providers(self):
        """Initialize provider instances with API keys"""

        self.update_status("Initializing providers...", self.COLORS["system"])

        # Check and initialize OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                self.providers["openai"] = OpenAIProvider(openai_key)
                self.append_system_message("✓ OpenAI initialized")
            except Exception as e:
                self.append_error_message(f"OpenAI initialization failed: {e}")
        else:
            self.append_system_message("⚠ OPENAI_API_KEY not set")

        # Check and initialize Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                self.providers["gemini"] = GeminiProvider(gemini_key)
                self.append_system_message("✓ Gemini initialized")
            except Exception as e:
                self.append_error_message(f"Gemini initialization failed: {e}")
        else:
            self.append_system_message("⚠ GEMINI_API_KEY not set")

        # Check and initialize Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(anthropic_key)
                self.append_system_message("✓ Anthropic initialized")
            except Exception as e:
                self.append_error_message(f"Anthropic initialization failed: {e}")
        else:
            self.append_system_message("⚠ ANTHROPIC_API_KEY not set")

        self.update_status("Ready")

    def on_provider_changed(self, event):
        """Update model list when provider changes"""
        provider = self.provider_var.get()
        models = self.provider_models.get(provider, [])
        self.model_combo["values"] = models
        if models:
            self.model_var.set(models[0])
            # Check if the new default model has requirements
            self.check_model_requirements(models[0])

        self.update_status(f"Switched to {provider.title()}")

    def on_model_changed(self, event):
        """Handle model selection changes"""
        model = self.model_var.get()
        self.check_model_requirements(model)

    def check_model_requirements(self, model: str):
        """Check if model has specific requirements and apply them"""
        if model in self.model_requirements:
            requirements = self.model_requirements[model]

            # Apply temperature requirement
            if "temperature" in requirements:
                required_temp = requirements["temperature"]
                current_temp = self.temp_var.get()

                if current_temp != required_temp:
                    self.temp_var.set(required_temp)
                    reason = requirements.get(
                        "reason", f"{model} requires specific settings"
                    )

                    # Show warning message
                    warning_msg = (
                        f"⚠ {reason} - Temperature automatically set to {required_temp}"
                    )
                    self.append_system_message(warning_msg)
                    self.update_status(
                        f"Temperature adjusted for {model}", self.COLORS["warning"]
                    )

                    # Highlight temperature label
                    self.temp_label.config(fg=self.COLORS["warning"])
        else:
            # Reset temperature label color if no requirements
            self.temp_label.config(fg=self.COLORS["text_secondary"])

    def append_message(self, role: str, content: str):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)

        # Add spacing before new message
        self.chat_display.insert(tk.END, "\n")

        # Add role header with timestamp
        timestamp = datetime.now().strftime("%H:%M")
        header = f"{role.upper()} · {timestamp}"
        self.chat_display.insert(tk.END, header + "\n", role)

        # Add message content
        self.chat_display.insert(tk.END, content + "\n", f"{role}_message")

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_system_message(self, content: str):
        """Add a system message"""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] {content}\n", "system")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_error_message(self, content: str):
        """Add an error message"""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ERROR: {content}\n", "error")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def clear_conversation(self):
        """Clear the conversation history"""
        if messagebox.askyesno("Clear Chat", "Clear conversation history?"):
            self.conversation_history.clear()
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.append_system_message("Conversation cleared")
            self.update_status("Ready")

    def attach_file(self):
        """Open file dialog to attach files"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.gif *.webp *.bmp"),
            ("Audio files", "*.mp3 *.wav"),
            ("Video files", "*.mp4 *.webm"),
            ("All files", "*.*"),
        ]
        filenames = filedialog.askopenfilenames(
            title="Select files to attach", filetypes=filetypes
        )
        if filenames:
            for filename in filenames:
                self.add_content_part("file", filename)

    def on_file_drop(self, event):
        """Handle drag-and-drop of files"""
        # Parse the dropped file paths
        files = self.root.tk.splitlist(event.data)
        for file_path in files:
            # Remove curly braces if present (Windows quirk)
            file_path = file_path.strip('{}')
            if os.path.isfile(file_path):
                self.add_content_part("file", file_path)
        return event.action

    def on_text_changed(self, event=None):
        """Handle text input changes to update preview"""
        # This is called on KeyRelease, so we don't update the preview live
        # as that would be too disruptive. We just update when sending.
        pass

    def add_content_part(self, part_type: str, value: str):
        """Add a content part (file or text) to the message"""
        part_id = str(uuid.uuid4())[:8]

        self.content_parts.append({
            "type": part_type,
            "value": value,
            "id": part_id
        })

        if part_type == "file" and value not in self.attached_files:
            self.attached_files.append(value)

        self.update_content_preview()
        self.update_status(
            f"{len(self.content_parts)} content part(s)", self.COLORS["success"]
        )

    def remove_content_part(self, part_id: str):
        """Remove a content part by ID"""
        # Find and remove the part
        for i, part in enumerate(self.content_parts):
            if part["id"] == part_id:
                # If it's a file, remove from attached_files too
                if part["type"] == "file" and part["value"] in self.attached_files:
                    self.attached_files.remove(part["value"])

                self.content_parts.pop(i)
                break

        self.update_content_preview()

        if self.content_parts:
            self.update_status(
                f"{len(self.content_parts)} content part(s)", self.COLORS["success"]
            )
        else:
            self.update_status("Ready")

    def move_content_part(self, part_id: str, direction: int):
        """Move a content part up (-1) or down (+1)"""
        for i, part in enumerate(self.content_parts):
            if part["id"] == part_id:
                new_index = i + direction
                if 0 <= new_index < len(self.content_parts):
                    self.content_parts[i], self.content_parts[new_index] = \
                        self.content_parts[new_index], self.content_parts[i]
                    self.update_content_preview()
                break

    def insert_text_marker(self):
        """Insert a text marker at the current position"""
        part_id = str(uuid.uuid4())[:8]

        self.content_parts.append({
            "type": "text_marker",
            "value": "[Your message text will go here]",
            "id": part_id
        })

        self.update_content_preview()

    def update_content_preview(self):
        """Update the visual display of content parts order"""
        # Clear existing preview
        for widget in self.content_preview_frame.winfo_children():
            widget.destroy()

        if not self.content_parts:
            # Hide the preview frame if no content parts
            self.content_preview_frame.pack_forget()
            return

        # Show the preview frame
        self.content_preview_frame.pack(fill=tk.X, pady=(0, 8))

        # Add header
        header_frame = tk.Frame(self.content_preview_frame, bg=self.COLORS["bg_tertiary"])
        header_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(
            header_frame,
            text="Message Structure (in order):",
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["text_secondary"],
            font=("Helvetica Neue", 10, "bold"),
        ).pack(side=tk.LEFT)

        # Add text marker button
        add_text_btn = tk.Button(
            header_frame,
            text="+ Insert Text Marker",
            command=self.insert_text_marker,
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["accent"],
            font=("Helvetica Neue", 9),
            relief="flat",
            borderwidth=0,
            padx=8,
            pady=2,
        )
        add_text_btn.pack(side=tk.RIGHT)

        # Add parts container
        parts_container = tk.Frame(self.content_preview_frame, bg=self.COLORS["bg_tertiary"])
        parts_container.pack(fill=tk.X, padx=8, pady=(0, 8))

        # Create content part items
        for idx, part in enumerate(self.content_parts):
            part_frame = tk.Frame(
                parts_container,
                bg=self.COLORS["bg_primary"],
                highlightbackground=self.COLORS["border"],
                highlightthickness=1,
            )
            part_frame.pack(fill=tk.X, pady=2)

            # Order number
            order_label = tk.Label(
                part_frame,
                text=f"{idx + 1}.",
                bg=self.COLORS["bg_primary"],
                fg=self.COLORS["text_dim"],
                font=("Helvetica Neue", 10, "bold"),
                padx=8,
            )
            order_label.pack(side=tk.LEFT)

            # Content based on type
            if part["type"] == "file":
                file_name = Path(part["value"]).name
                file_ext = Path(part["value"]).suffix.lower()

                # Icon
                if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
                    icon = "🖼️"
                elif file_ext in ['.mp3', '.wav']:
                    icon = "🎵"
                elif file_ext in ['.mp4', '.webm']:
                    icon = "🎬"
                else:
                    icon = "📄"

                content_label = tk.Label(
                    part_frame,
                    text=f"{icon} {file_name}",
                    bg=self.COLORS["bg_primary"],
                    fg=self.COLORS["text_primary"],
                    font=("Helvetica Neue", 10),
                    padx=4,
                )
                content_label.pack(side=tk.LEFT)

            elif part["type"] == "text_marker":
                content_label = tk.Label(
                    part_frame,
                    text="📝 [Your text message]",
                    bg=self.COLORS["bg_primary"],
                    fg=self.COLORS["accent"],
                    font=("Helvetica Neue", 10, "italic"),
                    padx=4,
                )
                content_label.pack(side=tk.LEFT)

            # Spacer
            tk.Frame(part_frame, bg=self.COLORS["bg_primary"]).pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Move buttons
            if idx > 0:
                up_btn = tk.Label(
                    part_frame,
                    text="▲",
                    bg=self.COLORS["bg_primary"],
                    fg=self.COLORS["text_secondary"],
                    font=("Helvetica Neue", 8),
                    padx=4,
                    cursor="hand2",
                )
                up_btn.pack(side=tk.RIGHT, padx=2)
                up_btn.bind("<Button-1>", lambda e, pid=part["id"]: self.move_content_part(pid, -1))

            if idx < len(self.content_parts) - 1:
                down_btn = tk.Label(
                    part_frame,
                    text="▼",
                    bg=self.COLORS["bg_primary"],
                    fg=self.COLORS["text_secondary"],
                    font=("Helvetica Neue", 8),
                    padx=4,
                    cursor="hand2",
                )
                down_btn.pack(side=tk.RIGHT, padx=2)
                down_btn.bind("<Button-1>", lambda e, pid=part["id"]: self.move_content_part(pid, 1))

            # Remove button
            remove_btn = tk.Label(
                part_frame,
                text="✕",
                bg=self.COLORS["bg_primary"],
                fg=self.COLORS["text_secondary"],
                font=("Helvetica Neue", 10, "bold"),
                padx=6,
                cursor="hand2",
            )
            remove_btn.pack(side=tk.RIGHT)
            remove_btn.bind("<Button-1>", lambda e, pid=part["id"]: self.remove_content_part(pid))

            # Hover effects
            def on_remove_enter(e, btn=remove_btn):
                btn.config(fg=self.COLORS["error"])

            def on_remove_leave(e, btn=remove_btn):
                btn.config(fg=self.COLORS["text_secondary"])

            remove_btn.bind("<Enter>", on_remove_enter)
            remove_btn.bind("<Leave>", on_remove_leave)

    def send_message(self):
        """Send the user's message and get LLM response"""
        if self.is_waiting:
            return

        # Get user input
        user_message = self.input_text.get(1.0, tk.END).strip()

        if not user_message and not self.content_parts:
            return

        # Clear input
        self.input_text.delete(1.0, tk.END)

        # Get selected provider and model
        provider_name = self.provider_var.get()
        model_name = self.model_var.get()
        temperature = self.temp_var.get()

        # Check if provider is available
        if provider_name not in self.providers:
            self.append_error_message(f"Provider '{provider_name}' not initialized")
            self.update_status("Error: Provider not initialized", self.COLORS["error"])
            return

        # Build message content based on content_parts structure
        llm_content_parts = []
        display_parts = []

        if self.content_parts:
            # Use the ordered content parts
            for part in self.content_parts:
                if part["type"] == "file":
                    # Process file
                    file_path = part["value"]
                    path = Path(file_path)
                    mime_type, _ = mimetypes.guess_type(str(path))

                    if not mime_type:
                        ext = path.suffix.lower()
                        mime_map = {
                            '.png': 'image/png',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.gif': 'image/gif',
                            '.webp': 'image/webp',
                            '.bmp': 'image/bmp',
                            '.mp3': 'audio/mpeg',
                            '.wav': 'audio/wav',
                            '.mp4': 'video/mp4',
                            '.webm': 'video/webm',
                        }
                        mime_type = mime_map.get(ext, 'application/octet-stream')

                    # Read and encode the file
                    with open(path, 'rb') as f:
                        file_data = f.read()
                        base64_data = base64.b64encode(file_data).decode('utf-8')

                    llm_content_parts.append(ContentPart(
                        type="image_base64",
                        data=base64_data,
                        mime_type=mime_type
                    ))
                    display_parts.append(f"[{path.name}]")

                elif part["type"] == "text_marker":
                    # Replace text marker with actual user message
                    if user_message:
                        llm_content_parts.append(ContentPart(type="text", text=user_message))
                        display_parts.append(user_message)

            # If no text marker was in the parts, but user has text, append it
            has_text_marker = any(p["type"] == "text_marker" for p in self.content_parts)
            if user_message and not has_text_marker:
                llm_content_parts.append(ContentPart(type="text", text=user_message))
                display_parts.append(user_message)

            display_message = "\n".join(display_parts)

            # Clear content parts
            self.content_parts.clear()
            self.attached_files.clear()
            self.update_content_preview()

        else:
            # Simple text message without any structure
            if not user_message:
                return

            llm_content_parts = [ContentPart(type="text", text=user_message)]
            display_message = user_message

        # Add to conversation history
        self.conversation_history.append(
            LLMMessage(role="user", content=llm_content_parts)
        )

        # Set waiting state
        self.is_waiting = True
        self.send_btn.config(
            text="Waiting...",
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["text_dim"],
            state=tk.DISABLED,
        )
        self.update_status(
            f"Sending to {provider_name} ({model_name})...", self.COLORS["system"]
        )

        # Add user message to display
        self.append_message("user", display_message)

        # Update UI
        self.root.update()

        # Make async call
        try:
            self.update_status(
                f"Waiting for {model_name} response...", self.COLORS["warning"]
            )
            response = asyncio.run(
                self.call_llm(provider_name, model_name, temperature)
            )

            # Add assistant response
            self.update_status(
                "Received response, processing...", self.COLORS["system"]
            )
            self.append_message("assistant", response.content)
            self.conversation_history.append(
                LLMMessage(role="assistant", content=response.content)
            )

            # Show token usage
            usage = f"Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})"
            self.append_system_message(usage)

            self.update_status("Ready", self.COLORS["success"])

        except Exception as e:
            self.append_error_message(f"Request failed: {str(e)}")
            self.update_status("Error occurred", self.COLORS["error"])

        finally:
            # Reset waiting state
            self.is_waiting = False
            self.send_btn.config(
                text="Send",
                bg=self.COLORS["bg_secondary"],
                fg=self.COLORS["text_primary"],
                state=tk.NORMAL,
            )

    async def call_llm(self, provider_name: str, model_name: str, temperature: float):
        """Call the LLM provider with current conversation history"""
        provider = self.providers[provider_name]

        response = await provider.call(
            model=model_name,
            messages=self.conversation_history,
            temperature=temperature,
        )

        return response


def main():
    """Main entry point"""

    # Check for API keys
    if (
        not os.getenv("OPENAI_API_KEY")
        and not os.getenv("GEMINI_API_KEY")
        and not os.getenv("ANTHROPIC_API_KEY")
    ):
        print("⚠  WARNING: No API keys found!")
        print("\nPlease set at least one environment variable:")
        print("  • OPENAI_API_KEY")
        print("  • GEMINI_API_KEY")
        print("  • ANTHROPIC_API_KEY")
        print("\nExample:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  python chat_gui.py\n")
        input("Press Enter to continue...")

    # Create and run GUI with drag-and-drop support
    root = TkinterDnD.Tk()
    app = ModernChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
