"""
Setup script for creating macOS application bundle for Platypus Chat.

This script uses py2app to create a standalone macOS application from chat_gui.py.
The resulting app will include all dependencies and can be distributed as a .app bundle.
"""

from setuptools import setup
import os

APP = ['chat_gui.py']
DATA_FILES = [
    ('', ['platypus.png']),  # Include logo in app bundle
    ('providers', [
        'providers/__init__.py',
        'providers/base.py',
        'providers/openai_provider.py',
        'providers/gemini_provider.py',
        'providers/anthropic_provider.py',
    ]),
]

OPTIONS = {
    'argv_emulation': False,  # Disable argv emulation for cleaner startup
    'iconfile': 'platypus.icns',  # Will be created from platypus.png
    'plist': {
        'CFBundleName': 'Platypus Chat',
        'CFBundleDisplayName': 'Platypus Chat',
        'CFBundleIdentifier': 'com.jimdex.platypus-chat',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13.0',
        'CFBundleDocumentTypes': [],
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
    },
    'packages': [
        'tkinter',
        'PIL',
        'httpx',
        'pydantic',
        'asyncio',
        'mimetypes',
        'base64',
        'uuid',
        'datetime',
        'pathlib',
    ],
    'includes': [
        'tkinterdnd2',
        'providers',
        'providers.base',
        'providers.openai_provider',
        'providers.gemini_provider',
        'providers.anthropic_provider',
    ],
    'resources': ['platypus.png'],
    'frameworks': [],
    'semi_standalone': False,  # Create fully standalone app
    'site_packages': True,
}

setup(
    name='Platypus Chat',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
