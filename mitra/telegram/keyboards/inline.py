"""
🤖 Mitra AI - Inline Keyboard Builders
Utilities for building inline keyboards.
Coded by Denvil with love 🤍
"""

from typing import List, Optional, Tuple
from telegram import InlineKeyboardMarkup, InlineKeyboardButton


class InlineKeyboardBuilder:
    """Builder for creating inline keyboards."""

    def __init__(self) -> None:
        self._rows: List[List[InlineKeyboardButton]] = []
        self._current_row: List[InlineKeyboardButton] = []

    def add_button(
        self,
        text: str,
        callback_data: Optional[str] = None,
        url: Optional[str] = None,
        switch_inline_query: Optional[str] = None,
        switch_inline_query_current_chat: Optional[str] = None,
    ) -> "InlineKeyboardBuilder":
        """Add a button to the current row."""
        button = InlineKeyboardButton(
            text=text,
            callback_data=callback_data,
            url=url,
            switch_inline_query=switch_inline_query,
            switch_inline_query_current_chat=switch_inline_query_current_chat,
        )
        self._current_row.append(button)
        return self

    def new_row(self) -> "InlineKeyboardBuilder":
        """Start a new row."""
        if self._current_row:
            self._rows.append(self._current_row)
            self._current_row = []
        return self

    def build(self) -> InlineKeyboardMarkup:
        """Build and return the keyboard."""
        # Add any remaining buttons
        if self._current_row:
            self._rows.append(self._current_row)
        return InlineKeyboardMarkup(self._rows)


class SettingsKeyboard:
    """Keyboards for settings menu."""

    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        """Get the main settings menu keyboard."""
        return (
            InlineKeyboardBuilder()
            .add_button("🔔 Notifications", callback_data="settings:notifications")
            .add_button("🌐 Language", callback_data="settings:language")
            .new_row()
            .add_button("🧠 Thinking Mode", callback_data="thinking_mode:select")
            .add_button("🎨 Theme", callback_data="settings:theme")
            .new_row()
            .add_button("📊 Privacy", callback_data="settings:privacy")
            .add_button("❌ Close", callback_data="cancel:")
            .build()
        )

    @staticmethod
    def notifications() -> InlineKeyboardMarkup:
        """Get notifications settings keyboard."""
        return (
            InlineKeyboardBuilder()
            .add_button("✅ Enable All", callback_data="settings:notifications_all")
            .add_button("🔕 Disable All", callback_data="settings:notifications_none")
            .new_row()
            .add_button("⬅️ Back", callback_data="settings:main")
            .build()
        )

    @staticmethod
    def language() -> InlineKeyboardMarkup:
        """Get language settings keyboard."""
        return (
            InlineKeyboardBuilder()
            .add_button("🇺🇸 English", callback_data="settings:lang_en")
            .add_button("🇷🇺 Русский", callback_data="settings:lang_ru")
            .new_row()
            .add_button("🇪🇸 Español", callback_data="settings:lang_es")
            .add_button("🇩🇪 Deutsch", callback_data="settings:lang_de")
            .new_row()
            .add_button("⬅️ Back", callback_data="settings:main")
            .build()
        )


class ThinkingModeKeyboard:
    """Keyboards for thinking mode selection."""

    @staticmethod
    def get_keyboard() -> InlineKeyboardMarkup:
        """Get thinking mode selection keyboard."""
        return (
            InlineKeyboardBuilder()
            .add_button("⚡ Instant", callback_data="thinking_mode:instant")
            .add_button("🔵 Standard", callback_data="thinking_mode:standard")
            .new_row()
            .add_button("🧠 Deep", callback_data="thinking_mode:deep")
            .add_button("🎓 Expert", callback_data="thinking_mode:expert")
            .new_row()
            .add_button("🔮 Maximum", callback_data="thinking_mode:maximum")
            .new_row()
            .add_button("⬅️ Back", callback_data="settings:main")
            .build()
        )


class ConfirmationKeyboard:
    """Keyboards for confirmation dialogs."""

    @staticmethod
    def yes_no(action_id: str = "") -> InlineKeyboardMarkup:
        """Get a yes/no confirmation keyboard."""
        return (
            InlineKeyboardBuilder()
            .add_button("✅ Yes", callback_data=f"confirm:{action_id}")
            .add_button("❌ No", callback_data=f"cancel:{action_id}")
            .build()
        )

    @staticmethod
    def delete_confirm(item_id: str = "") -> InlineKeyboardMarkup:
        """Get a delete confirmation keyboard."""
        return (
            InlineKeyboardBuilder()
            .add_button("🗑️ Delete", callback_data=f"delete_confirm:{item_id}")
            .add_button("⬅️ Cancel", callback_data=f"cancel:{item_id}")
            .build()
        )


class PaginationKeyboard:
    """Keyboards for paginated content."""

    @staticmethod
    def get_keyboard(
        current_page: int,
        total_pages: int,
        prefix: str = "page",
    ) -> InlineKeyboardMarkup:
        """Get a pagination keyboard."""
        builder = InlineKeyboardBuilder()

        # Previous button
        if current_page > 1:
            builder.add_button("⬅️", callback_data=f"{prefix}:{current_page - 1}")
        else:
            builder.add_button("⬅️", callback_data=f"{prefix}:1")

        # Page indicator
        builder.add_button(f"{current_page}/{total_pages}", callback_data=f"{prefix}:info")

        # Next button
        if current_page < total_pages:
            builder.add_button("➡️", callback_data=f"{prefix}:{current_page + 1}")
        else:
            builder.add_button("➡️", callback_data=f"{prefix}:{total_pages}")

        return builder.build()

    @staticmethod
    def with_items(
        items: List[Tuple[str, str]],
        current_page: int,
        total_pages: int,
        items_per_row: int = 2,
        prefix: str = "page",
    ) -> InlineKeyboardMarkup:
        """Get a pagination keyboard with item buttons."""
        builder = InlineKeyboardBuilder()

        # Add item buttons
        for i, (text, callback) in enumerate(items):
            builder.add_button(text, callback_data=callback)
            if (i + 1) % items_per_row == 0:
                builder.new_row()

        # Add navigation row
        builder.new_row()

        # Previous button
        if current_page > 1:
            builder.add_button("⬅️", callback_data=f"{prefix}:{current_page - 1}")

        # Page indicator
        builder.add_button(f"{current_page}/{total_pages}", callback_data=f"{prefix}:info")

        # Next button
        if current_page < total_pages:
            builder.add_button("➡️", callback_data=f"{prefix}:{current_page + 1}")

        return builder.build()
