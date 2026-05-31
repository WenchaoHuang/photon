/**
 *	Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in all
 *	copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *	SOFTWARE.
 * 
 *	Repo URL: https://github.com/WenchaoHuang/easywin32.git
 */
#pragma once

// C headers
#include <assert.h>
#include <Windows.h>
#include <dwmapi.h>

// C++ headers
#include <string>
#include <vector>
#include <functional>

// Link dwmapi.lib
#pragma comment(lib, "dwmapi.lib")

/*********************************************************************************
********************************    EasyWin32    *********************************
*********************************************************************************/

namespace easywin32
{
	class Window;

	using Byte = BYTE;
	using Size = SIZE;
	using Rect = RECT;
	using Point = POINT;
	using Result = LRESULT;
#ifdef UNICODE
	using string_type = std::wstring;
#else
	using string_type = std::string;
#endif

	/*****************************************************************************
	********************************    Color    *********************************
	*****************************************************************************/

	//!	@brief	Represents an RGB color with 8-bit channels.
	struct ColorRGB
	{
		uint8_t		r;	//!< Red channel component (0–255)
		uint8_t		g;	//!< Green channel component (0–255)
		uint8_t		b;	//!< Blue channel component (0–255)
	};

	//!	@brief	Represents an RGBA color with 8-bit channels.
	struct ColorRGBA : public ColorRGB
	{
		uint8_t		a;	//!< Alpha channel component (0–255)
	};

	/*****************************************************************************
	***************************    Flags<EnumType>    ****************************
	*****************************************************************************/

	//!	@brief		A strongly-typed bit flag wrapper for enum classes.
	template<typename EnumType> struct Flags
	{
		static_assert(std::is_enum_v<EnumType>, "Flags<T>: T must be an enum type.");

		using Type = std::underlying_type_t<EnumType>;

		Type	mask;

		// Constructors
		constexpr Flags() noexcept : mask(0) {}
		constexpr Flags(Type value) noexcept : mask(value) {}
		constexpr Flags(EnumType value) noexcept : mask(static_cast<Type>(value)) {}

		// Bitwise OR
		constexpr Flags operator|(Flags rhs) const noexcept { return Flags(mask | rhs.mask); }
		constexpr Flags operator|(EnumType rhs) const noexcept { return Flags(mask | static_cast<Type>(rhs)); }
		constexpr Flags & operator|=(EnumType rhs) noexcept { mask |= static_cast<Type>(rhs); return *this; }
		constexpr Flags & operator|=(Flags rhs) noexcept { mask |= rhs.mask; return *this; }

		// Bitwise AND
		constexpr Flags operator&(Flags rhs) const noexcept { return Flags(mask & rhs.mask); }
		constexpr Flags operator&(EnumType rhs) const noexcept { return Flags(mask & static_cast<Type>(rhs)); }
		constexpr Flags & operator&=(EnumType rhs) noexcept { mask &= static_cast<Type>(rhs); return *this; }
		constexpr Flags & operator&=(Flags rhs) noexcept { mask &= rhs.mask; return *this; }

		// Bitwise XOR
		constexpr Flags operator^(Flags rhs) const noexcept { return Flags(mask ^ rhs.mask); }
		constexpr Flags & operator^=(Flags rhs) noexcept { mask ^= rhs.mask; return *this; }

		// Bitwise NOT
		constexpr Flags operator~() const noexcept { return Flags(~mask); }

		// Conversion
		constexpr operator Type() const noexcept { return mask; }

		// Test
		constexpr bool has(Flags flags) const noexcept { return flags ? (flags.mask & mask) == flags.mask : !flags; }
		constexpr bool none() const noexcept { return mask == 0; }
		constexpr bool any() const noexcept { return mask != 0; }
	};

	/*****************************************************************************
	**********************    EZWIN32_ENABLE_ENUM_FLAGS    ***********************
	*****************************************************************************/

	/**
	 *	@brief		Enables bitwise operators (|, &, ~) for a strongly-typed enum.
	 *	@details	This macro defines overloaded operators that allow using the enum as
	 *				bit flags through the `Flags<EnumType>` wrapper.
	 */
	#define EZWIN32_ENABLE_ENUM_FLAGS(EnumType)									\
																				\
	constexpr Flags<EnumType> operator|(EnumType lhs, EnumType rhs) noexcept	\
	{																			\
		return Flags<EnumType>(lhs) | Flags<EnumType>(rhs);						\
	}																			\
	constexpr Flags<EnumType> operator&(EnumType lhs, EnumType rhs) noexcept	\
	{																			\
		return Flags<EnumType>(lhs) & Flags<EnumType>(rhs);						\
	}																			\
	constexpr Flags<EnumType> operator~(EnumType v) noexcept					\
	{																			\
		return ~Flags<EnumType>(v);												\
	}

	/*****************************************************************************
	*************************    EZWIN32_CASE_TO_STR    **************************
	*****************************************************************************/

	/**
	 *	@brief		Macro to convert an enum value to its string representation in a switch case.
	 *	@details	Simplifies switch statements by generating a case branch that returns the stringified
	 *				form of the enum value (e.g., `Key::A` to `"Key::A"`). Used in `to_string` functions
	 *				for debugging or logging purposes.
	 * @param[in]	x - The enum value to convert to a string.
	 */
	#define EZWIN32_CASE_TO_STR(x)					case x: return #x

	/**
	 *	@brief		Appends the name of a flag (x) to a string if that flag is set.
	 *	@details	This macro is typically used when you want to convert a bitmask or flag set into a readable string.
	 */
	#define EZWIN32_CASE_APPEND_STR(result, flags, x)							\
																				\
		if (flags.has(x))														\
		{																		\
			if (!result.empty())												\
			{																	\
				result += " | ";												\
			}																	\
																				\
			result += #x;														\
		}

	/*****************************************************************************
	********************************    Style    *********************************
	*****************************************************************************/

	/**
	 *	@brief	Window styles
	 *	@see	https://learn.microsoft.com/en-us/windows/win32/winmsg/window-styles
	 */
	enum class Style
	{
		Border				= 1 << 0,							//!< Thin border
		Visible				= 1 << 1,							//!< Initially visible
		SysMenu				= 1 << 2,							//!< System menu (requires `Caption`)
		Disabled			= 1 << 3,							//!< Initially disabled
		ThickFrame			= 1 << 5,							//!< Thick border
		MinimizeBox			= 1 << 6,							//!< Minimize button (requires `SysMenu`)
		MaximizeBox			= 1 << 7,							//!< Maximize button (requires `SysMenu`)
		Resizable			= ThickFrame | 1 << 4,				//!< Resizable border (include `ThickFrame`)
		Caption				= Border | ThickFrame | (1 << 8),	//!< Title bar (include `Border` + `ThickFrame`)

		// Combined styles
		PopupWindow			= SysMenu | Border,
		HeaderlessWindow	= SysMenu | Resizable | MaximizeBox | MinimizeBox,
		OverlappedWindow	= SysMenu | Resizable | MaximizeBox | MinimizeBox | Caption,
	};

	//!	@brief	Enables bitwise operators (|, &, ~).
	EZWIN32_ENABLE_ENUM_FLAGS(Style);

	//!	@brief	Converts a `Flags<Style>` to a readable string.
	static inline std::string to_string(Flags<Style> styleFlags)
	{
		std::string result;

		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::Border);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::Visible);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::SysMenu);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::Caption);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::Disabled);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::Resizable);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::ThickFrame);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::MinimizeBox);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::MaximizeBox);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::PopupWindow);
		EZWIN32_CASE_APPEND_STR(result, styleFlags, Style::OverlappedWindow);

		return result.empty() ?	"Style::Empty" : result;
	}

	/*****************************************************************************
	*******************************    ExStyle    ********************************
	*****************************************************************************/

	/**
	 *	@brief	Extended window styles.
	 *	@see	https://learn.microsoft.com/en-us/windows/win32/winmsg/extended-window-styles
	 */
	enum class ExStyle : DWORD
	{
		// Individual styles
		AcceptFiles				= WS_EX_ACCEPTFILES,			//!< Accepts drag-drop files.
		AppWindow				= WS_EX_APPWINDOW,				//!< Forces top-level window onto taskbar when visible.
		Composited				= WS_EX_COMPOSITED,				//!< Paints descendants bottom-to-top with double-buffering.
		ContextHelp				= WS_EX_CONTEXTHELP,			//!< Title bar question mark for help.
		ControlParent			= WS_EX_CONTROLPARENT,			//!< Enables dialog navigation for child windows.
		Layered					= WS_EX_LAYERED,				//!< Layered window, supports top-level and child windows (Windows 8+).
		LayoutRtl				= WS_EX_LAYOUTRTL,				//!< Right-edge origin for right-to-left languages.
		Left					= WS_EX_LEFT,					//!< Generic left-aligned properties (default).
		LtrReading				= WS_EX_LTRREADING,				//!< Left-to-right text reading order (default).
		MDIChild				= WS_EX_MDICHILD,				//!< MDI child window.
		NoActivate				= WS_EX_NOACTIVATE,				//!< Non-foreground window, not on taskbar by default.
		NoInheritLayout			= WS_EX_NOINHERITLAYOUT,		//!< Does not pass layout to child windows.
		NoParentNotify			= WS_EX_NOPARENTNOTIFY,			//!< No WM_PARENTNOTIFY for child creation/destruction.
		NoRedirectionBitmap		= WS_EX_NOREDIRECTIONBITMAP,	//!< No redirection surface rendering.
		Right					= WS_EX_RIGHT,					//!< Right-aligned properties for right-to-left languages.
		RtlReading				= WS_EX_RTLREADING,				//!< Right-to-left text reading order for supported languages.
		ToolWindow				= WS_EX_TOOLWINDOW,				//!< Floating toolbar, no taskbar/ALT+TAB.
		TopMost					= WS_EX_TOPMOST,				//!< Stays above non-topmost windows.
		Transparent				= WS_EX_TRANSPARENT,			//!< Delays painting until siblings are painted.
		// Deprecated (ugly)
		StaticEdge				= WS_EX_STATICEDGE,				//!< 3D border for non-interactive items.
		WindowEdge				= WS_EX_WINDOWEDGE,				//!< Border with raised edge.
		ClientEdge				= WS_EX_CLIENTEDGE,				//!< Border with sunken edge.
		DlgModalFrame			= WS_EX_DLGMODALFRAME,			//!< Double border, optional title bar with WS_CAPTION.
		LeftScrollBar			= WS_EX_LEFTSCROLLBAR,			//!< Left-side scroll bar for right-to-left languages.
		RightScrollBar			= WS_EX_RIGHTSCROLLBAR,			//!< Right-side scroll bar (default).
		
		// Combined styles
		OverlappedWindow		= WS_EX_OVERLAPPEDWINDOW,		//!< Overlapped window (WINDOWEDGE | CLIENTEDGE).
		PaletteWindow			= WS_EX_PALETTEWINDOW,			//!< Palette window (WINDOWEDGE | TOOLWINDOW | TOPMOST).
	};

	//!	@brief	Enables bitwise operators (|, &, ~).
	EZWIN32_ENABLE_ENUM_FLAGS(ExStyle);

	//!	@brief	Converts a `Flags<ExStyle>` to a readable string.
	static inline std::string to_string(Flags<ExStyle> exStyleFlags)
	{
		std::string result;

		// Individual
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::AcceptFiles);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::AppWindow);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::ClientEdge);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::Composited);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::ContextHelp);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::ControlParent);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::DlgModalFrame);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::Layered);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::LayoutRtl);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::LeftScrollBar);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::LtrReading);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::MDIChild);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::NoActivate);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::NoInheritLayout);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::NoParentNotify);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::NoRedirectionBitmap);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::Right);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::RightScrollBar);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::RtlReading);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::StaticEdge);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::ToolWindow);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::TopMost);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::Transparent);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::WindowEdge);
		// Combined
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::OverlappedWindow);
		EZWIN32_CASE_APPEND_STR(result, exStyleFlags, ExStyle::PaletteWindow);

		return result.empty() ? "ExStyle::Left" : result;
	}

	/*****************************************************************************
	******************************    CornerStyle    *****************************
	*****************************************************************************/

	//!	@brief	Corner style of the window.
	enum class CornerStyle
	{
		Default				= DWMWCP_DEFAULT,			//!< Let the system decide whether or not to round window corners
		DoNotRound			= DWMWCP_DONOTROUND,		//!< Never round window corners
		Round				= DWMWCP_ROUND,				//!< Round the corners if appropriate
		RoundSmall			= DWMWCP_ROUNDSMALL,		//!< Round the corners if appropriate, with a small radius
	};

	/*****************************************************************************
	*******************************    Backdrop    *******************************
	*****************************************************************************/
	
	//!	@brief	System backdrop types.
	enum class Backdrop
	{
		Auto				= DWMSBT_AUTO,				//!< [Default] Let DWM automatically decide the system-drawn backdrop for this window.
		None				= DWMSBT_NONE,				//!< Do not draw any system backdrop.
		MainWindow			= DWMSBT_MAINWINDOW,		//!< Draw the backdrop material effect corresponding to a long-lived window.
		TabbedWindow		= DWMSBT_TABBEDWINDOW,		//!< Draw the backdrop material effect corresponding to a window with a tabbed title bar.
		TransientWindow		= DWMSBT_TRANSIENTWINDOW,	//!< Draw the backdrop material effect corresponding to a transient window.
	};

	/*****************************************************************************
	***********************    NonClientRenderingPolicy    ***********************
	*****************************************************************************/

	//!	@brief	Non-client rendering policy values.
	enum class NonClientRenderingPolicy
	{
		UseWindowStyle		= DWMNCRP_USEWINDOWSTYLE,	//!< Enable/disable non-client rendering based on window style
		Disabled			= DWMNCRP_DISABLED,			//!< Disabled non-client rendering; window style is ignored
		Enabled				= DWMNCRP_ENABLED,			//!< Enabled non-client rendering; window style is ignored
	};
	
	/*****************************************************************************
	*********************************    Key    **********************************
	*****************************************************************************/

	//!	@brief	Virtual-key codes used by the system.
	enum class Key
	{
		A = 'A',
		B = 'B',
		C = 'C',
		D = 'D',
		E = 'E',
		F = 'F',
		G = 'G',
		H = 'H',
		I = 'I',
		J = 'J',
		K = 'K',
		L = 'L',
		M = 'M',
		N = 'N',
		O = 'O',
		P = 'P',
		Q = 'Q',
		R = 'R',
		S = 'S',
		T = 'T',
		U = 'U',
		V = 'V',
		W = 'W',
		X = 'X',
		Y = 'Y',
		Z = 'Z',

		Num0 = '0',
		Num1 = '1',
		Num2 = '2',
		Num3 = '3',
		Num4 = '4',
		Num5 = '5',
		Num6 = '6',
		Num7 = '7',
		Num8 = '8',
		Num9 = '9',

		F1 = VK_F1,
		F2 = VK_F2,
		F3 = VK_F3,
		F4 = VK_F4,
		F5 = VK_F5,
		F6 = VK_F6,
		F7 = VK_F7,
		F8 = VK_F8,
		F9 = VK_F9,
		F10 = VK_F10,
		F11 = VK_F11,
		F12 = VK_F12,
		F13 = VK_F13,
		F14 = VK_F14,
		F15 = VK_F15,
		F16 = VK_F16,
		F17 = VK_F17,
		F18 = VK_F18,
		F19 = VK_F19,
		F20 = VK_F20,
		F21 = VK_F21,
		F22 = VK_F22,
		F23 = VK_F23,
		F24 = VK_F24,

		Pause = VK_PAUSE,
		ScrollLock = VK_SCROLL,
		PrintScreen = VK_SNAPSHOT,

		End = VK_END,
		Home = VK_HOME,
		Insert = VK_INSERT,
		Delete = VK_DELETE,
		PageUp = VK_PRIOR,
		PageDown = VK_NEXT,

		Up = VK_UP,
		Down = VK_DOWN,
		Left = VK_LEFT,
		Right = VK_RIGHT,

		Tab = VK_TAB,
		Alt = VK_MENU,
		Apps = VK_APPS,
		Space = VK_SPACE,
		Clear = VK_CLEAR,
		Shift = VK_SHIFT,
		Enter = VK_RETURN,
		LeftWin = VK_LWIN,
		RightWin = VK_RWIN,
		Escape = VK_ESCAPE,
		Control = VK_CONTROL,
		CapLock = VK_CAPITAL,
		BackSpace = VK_BACK,

		NumLock = VK_NUMLOCK,
		NumPad0 = VK_NUMPAD0,
		NumPad1 = VK_NUMPAD1,
		NumPad2 = VK_NUMPAD2,
		NumPad3 = VK_NUMPAD3,
		NumPad4 = VK_NUMPAD4,
		NumPad5 = VK_NUMPAD5,
		NumPad6 = VK_NUMPAD6,
		NumPad7 = VK_NUMPAD7,
		NumPad8 = VK_NUMPAD8,
		NumPad9 = VK_NUMPAD9,

		Add = VK_ADD,
		Divide = VK_DIVIDE,
		Decimal = VK_DECIMAL,
		Subtract = VK_SUBTRACT,
		Multiply = VK_MULTIPLY,

		VolumeUp = VK_VOLUME_UP,
		VolumeMute = VK_VOLUME_MUTE,
		VolumeDown = VK_VOLUME_DOWN,

		MediaStop = VK_MEDIA_STOP,
		MediaPlayPause = VK_MEDIA_PLAY_PAUSE,
		MediaNextTrack = VK_MEDIA_NEXT_TRACK,
		MediaPrevTrack = VK_MEDIA_PREV_TRACK,

		LaunchMail = VK_LAUNCH_MAIL,
		LaunchApp1 = VK_LAUNCH_APP1,
		LaunchApp2 = VK_LAUNCH_APP2,
		LaunchMediaSelect = VK_LAUNCH_MEDIA_SELECT,

		BrowserHome = VK_BROWSER_HOME,
		BrowserStop = VK_BROWSER_STOP,
		BrowserBack = VK_BROWSER_BACK,
		BrowserSearch = VK_BROWSER_SEARCH,
		BrowserForward = VK_BROWSER_FORWARD,
		BrowserRefresh = VK_BROWSER_REFRESH,
		BrowserFavorites = VK_BROWSER_FAVORITES,
	};

	//! @brief	Convert `Key` enum to string for debugging or logging.
	static inline const char * to_string(Key key)
	{
		switch (key)
		{
			EZWIN32_CASE_TO_STR(Key::A);
			EZWIN32_CASE_TO_STR(Key::B);
			EZWIN32_CASE_TO_STR(Key::C);
			EZWIN32_CASE_TO_STR(Key::D);
			EZWIN32_CASE_TO_STR(Key::E);
			EZWIN32_CASE_TO_STR(Key::F);
			EZWIN32_CASE_TO_STR(Key::G);
			EZWIN32_CASE_TO_STR(Key::H);
			EZWIN32_CASE_TO_STR(Key::I);
			EZWIN32_CASE_TO_STR(Key::J);
			EZWIN32_CASE_TO_STR(Key::K);
			EZWIN32_CASE_TO_STR(Key::L);
			EZWIN32_CASE_TO_STR(Key::M);
			EZWIN32_CASE_TO_STR(Key::N);
			EZWIN32_CASE_TO_STR(Key::O);
			EZWIN32_CASE_TO_STR(Key::P);
			EZWIN32_CASE_TO_STR(Key::Q);
			EZWIN32_CASE_TO_STR(Key::R);
			EZWIN32_CASE_TO_STR(Key::S);
			EZWIN32_CASE_TO_STR(Key::T);
			EZWIN32_CASE_TO_STR(Key::U);
			EZWIN32_CASE_TO_STR(Key::V);
			EZWIN32_CASE_TO_STR(Key::W);
			EZWIN32_CASE_TO_STR(Key::X);
			EZWIN32_CASE_TO_STR(Key::Y);
			EZWIN32_CASE_TO_STR(Key::Z);

			EZWIN32_CASE_TO_STR(Key::Num0);
			EZWIN32_CASE_TO_STR(Key::Num1);
			EZWIN32_CASE_TO_STR(Key::Num2);
			EZWIN32_CASE_TO_STR(Key::Num3);
			EZWIN32_CASE_TO_STR(Key::Num4);
			EZWIN32_CASE_TO_STR(Key::Num5);
			EZWIN32_CASE_TO_STR(Key::Num6);
			EZWIN32_CASE_TO_STR(Key::Num7);
			EZWIN32_CASE_TO_STR(Key::Num8);
			EZWIN32_CASE_TO_STR(Key::Num9);

			EZWIN32_CASE_TO_STR(Key::F1);
			EZWIN32_CASE_TO_STR(Key::F2);
			EZWIN32_CASE_TO_STR(Key::F3);
			EZWIN32_CASE_TO_STR(Key::F4);
			EZWIN32_CASE_TO_STR(Key::F5);
			EZWIN32_CASE_TO_STR(Key::F6);
			EZWIN32_CASE_TO_STR(Key::F7);
			EZWIN32_CASE_TO_STR(Key::F8);
			EZWIN32_CASE_TO_STR(Key::F9);
			EZWIN32_CASE_TO_STR(Key::F10);
			EZWIN32_CASE_TO_STR(Key::F11);
			EZWIN32_CASE_TO_STR(Key::F12);
			EZWIN32_CASE_TO_STR(Key::F13);
			EZWIN32_CASE_TO_STR(Key::F14);
			EZWIN32_CASE_TO_STR(Key::F15);
			EZWIN32_CASE_TO_STR(Key::F16);
			EZWIN32_CASE_TO_STR(Key::F17);
			EZWIN32_CASE_TO_STR(Key::F18);
			EZWIN32_CASE_TO_STR(Key::F19);
			EZWIN32_CASE_TO_STR(Key::F20);
			EZWIN32_CASE_TO_STR(Key::F21);
			EZWIN32_CASE_TO_STR(Key::F22);
			EZWIN32_CASE_TO_STR(Key::F23);
			EZWIN32_CASE_TO_STR(Key::F24);

			EZWIN32_CASE_TO_STR(Key::Pause);
			EZWIN32_CASE_TO_STR(Key::ScrollLock);
			EZWIN32_CASE_TO_STR(Key::PrintScreen);

			EZWIN32_CASE_TO_STR(Key::End);
			EZWIN32_CASE_TO_STR(Key::Home);
			EZWIN32_CASE_TO_STR(Key::Insert);
			EZWIN32_CASE_TO_STR(Key::Delete);
			EZWIN32_CASE_TO_STR(Key::PageUp);
			EZWIN32_CASE_TO_STR(Key::PageDown);

			EZWIN32_CASE_TO_STR(Key::Up);
			EZWIN32_CASE_TO_STR(Key::Down);
			EZWIN32_CASE_TO_STR(Key::Left);
			EZWIN32_CASE_TO_STR(Key::Right);

			EZWIN32_CASE_TO_STR(Key::Tab);
			EZWIN32_CASE_TO_STR(Key::Alt);
			EZWIN32_CASE_TO_STR(Key::Apps);
			EZWIN32_CASE_TO_STR(Key::Space);
			EZWIN32_CASE_TO_STR(Key::Clear);
			EZWIN32_CASE_TO_STR(Key::Shift);
			EZWIN32_CASE_TO_STR(Key::Enter);
			EZWIN32_CASE_TO_STR(Key::LeftWin);
			EZWIN32_CASE_TO_STR(Key::RightWin);
			EZWIN32_CASE_TO_STR(Key::Escape);
			EZWIN32_CASE_TO_STR(Key::Control);
			EZWIN32_CASE_TO_STR(Key::CapLock);
			EZWIN32_CASE_TO_STR(Key::BackSpace);

			EZWIN32_CASE_TO_STR(Key::NumLock);
			EZWIN32_CASE_TO_STR(Key::NumPad0);
			EZWIN32_CASE_TO_STR(Key::NumPad1);
			EZWIN32_CASE_TO_STR(Key::NumPad2);
			EZWIN32_CASE_TO_STR(Key::NumPad3);
			EZWIN32_CASE_TO_STR(Key::NumPad4);
			EZWIN32_CASE_TO_STR(Key::NumPad5);
			EZWIN32_CASE_TO_STR(Key::NumPad6);
			EZWIN32_CASE_TO_STR(Key::NumPad7);
			EZWIN32_CASE_TO_STR(Key::NumPad8);
			EZWIN32_CASE_TO_STR(Key::NumPad9);

			EZWIN32_CASE_TO_STR(Key::Add);
			EZWIN32_CASE_TO_STR(Key::Divide);
			EZWIN32_CASE_TO_STR(Key::Decimal);
			EZWIN32_CASE_TO_STR(Key::Subtract);
			EZWIN32_CASE_TO_STR(Key::Multiply);

			EZWIN32_CASE_TO_STR(Key::VolumeUp);
			EZWIN32_CASE_TO_STR(Key::VolumeMute);
			EZWIN32_CASE_TO_STR(Key::VolumeDown);

			EZWIN32_CASE_TO_STR(Key::MediaStop);
			EZWIN32_CASE_TO_STR(Key::MediaPlayPause);
			EZWIN32_CASE_TO_STR(Key::MediaNextTrack);
			EZWIN32_CASE_TO_STR(Key::MediaPrevTrack);

			EZWIN32_CASE_TO_STR(Key::LaunchMail);
			EZWIN32_CASE_TO_STR(Key::LaunchApp1);
			EZWIN32_CASE_TO_STR(Key::LaunchApp2);
			EZWIN32_CASE_TO_STR(Key::LaunchMediaSelect);

			EZWIN32_CASE_TO_STR(Key::BrowserHome);
			EZWIN32_CASE_TO_STR(Key::BrowserStop);
			EZWIN32_CASE_TO_STR(Key::BrowserBack);
			EZWIN32_CASE_TO_STR(Key::BrowserSearch);
			EZWIN32_CASE_TO_STR(Key::BrowserForward);
			EZWIN32_CASE_TO_STR(Key::BrowserRefresh);
			EZWIN32_CASE_TO_STR(Key::BrowserFavorites);

			default:	return "Key::Unknown";
		}
	}

	/*****************************************************************************
	*****************************    MouseButton    ******************************
	*****************************************************************************/

	//!	@brief	Mouse button types.
	enum class MouseButton
	{
		Left,			//!< Left mouse button.
		Right,			//!< Right mouse button.
		Middle,			//!< Middle mouse button (usually the wheel).
		XButton1,		//!< Extra mouse button 1.
		XButton2,		//!< Extra mouse button 2.
	};

	//!	@brief	Converts a `MouseButton` enum value to a string representation.
	static inline const char * to_string(MouseButton button)
	{
		switch (button)
		{
			EZWIN32_CASE_TO_STR(MouseButton::Left);
			EZWIN32_CASE_TO_STR(MouseButton::Right);
			EZWIN32_CASE_TO_STR(MouseButton::Middle);
			EZWIN32_CASE_TO_STR(MouseButton::XButton1);
			EZWIN32_CASE_TO_STR(MouseButton::XButton2);
			default:	return "MouseButton::Unknown";
		}
	}

	/*****************************************************************************
	******************************    MouseState    ******************************
	*****************************************************************************/

	//! @brief	Mouse state bit flags representing the current button and key states.
	enum class MouseState
	{
		Left		= MK_LBUTTON,		//!< Left mouse button is pressed.
		Right		= MK_RBUTTON,		//!< Right mouse button is pressed.
		Middle		= MK_MBUTTON,		//!< Middle mouse button is pressed.
		XButton1	= MK_XBUTTON1,		//!< Extra mouse button 1 is pressed.
		XButton2	= MK_XBUTTON2,		//!< Extra mouse button 2 is pressed.
		Shift		= MK_SHIFT,			//!< SHIFT key is held down.
		Ctrl		= MK_CONTROL,		//!< CTRL key is held down.
	};

	//!	@brief	Enables bitwise operators (|, &, ~).
	EZWIN32_ENABLE_ENUM_FLAGS(MouseState);

	//!	@brief	Converts a `Flags<MouseState>` to a readable string.
	static inline std::string to_string(Flags<MouseState> stateFlags)
	{
		std::string result;

		EZWIN32_CASE_APPEND_STR(result, stateFlags, MouseState::Left);
		EZWIN32_CASE_APPEND_STR(result, stateFlags, MouseState::Right);
		EZWIN32_CASE_APPEND_STR(result, stateFlags, MouseState::Middle);
		EZWIN32_CASE_APPEND_STR(result, stateFlags, MouseState::XButton1);
		EZWIN32_CASE_APPEND_STR(result, stateFlags, MouseState::XButton2);
		EZWIN32_CASE_APPEND_STR(result, stateFlags, MouseState::Shift);
		EZWIN32_CASE_APPEND_STR(result, stateFlags, MouseState::Ctrl);

		return result.empty() ? "MouseState::None" : result;
	}

	/*****************************************************************************
	*****************************    MouseAction    ******************************
	*****************************************************************************/

	//! @brief	Type of mouse action.
	enum class MouseAction
	{
		Up,				//!< Mouse button released.
		Down,			//!< Mouse button pressed.
		DoubleClick,	//!< Mouse button double-clicked.
	};

	//! @brief	Converts a `MouseAction` enum value to a string representation.
	static inline const char * to_string(MouseAction action)
	{
		switch (action)
		{
			EZWIN32_CASE_TO_STR(MouseAction::Up);
			EZWIN32_CASE_TO_STR(MouseAction::Down);
			EZWIN32_CASE_TO_STR(MouseAction::DoubleClick);
			default:	return "MouseAction::Unknown";
		}
	}

	/*****************************************************************************
	******************************    KeyAction    *******************************
	*****************************************************************************/

	//!	@brief	Type of keyboard action.
	enum class KeyAction
	{
		Press,		//!< The key has been pressed down for the first time.
		Repeat,		//!< The key is being held down and generating repeat messages.
		Release,	//!< The key has been released.
	};

	//!	@brief	Convert `KeyAction` enum to string for debugging or logging.
	static inline const char * to_string(KeyAction action)
	{
		switch (action)
		{
			EZWIN32_CASE_TO_STR(KeyAction::Press);
			EZWIN32_CASE_TO_STR(KeyAction::Repeat);
			EZWIN32_CASE_TO_STR(KeyAction::Release);
			default:	return "KeyAction::Unknown";
		}
	}

	/*****************************************************************************
	********************************    Cursor    ********************************
	*****************************************************************************/

	//!	@brief	Predefined cursors of Windows OS.
	enum class Cursor : uint64_t
	{
		None			= (uint64_t)IDC_NO,
		Wait			= (uint64_t)IDC_WAIT,
		Hand			= (uint64_t)IDC_HAND,
		Help			= (uint64_t)IDC_HELP,
		Arrow			= (uint64_t)IDC_ARROW,
		Cross			= (uint64_t)IDC_CROSS,
		IBeam			= (uint64_t)IDC_IBEAM,
		SizeWE			= (uint64_t)IDC_SIZEWE,
		SizeNS			= (uint64_t)IDC_SIZENS,
		SizeAll			= (uint64_t)IDC_SIZEALL,
		UpArrow			= (uint64_t)IDC_UPARROW,
		SizeNWSE		= (uint64_t)IDC_SIZENWSE,
		SizeNESW		= (uint64_t)IDC_SIZENESW,
		AppStarting		= (uint64_t)IDC_APPSTARTING,
	};

	//!	@brief	Convert `Cursor` enum to string for debugging or logging.
	static inline const char* to_string(Cursor cursor)
	{
		switch (cursor)
		{
			EZWIN32_CASE_TO_STR(Cursor::None);
			EZWIN32_CASE_TO_STR(Cursor::Wait);
			EZWIN32_CASE_TO_STR(Cursor::Hand);
			EZWIN32_CASE_TO_STR(Cursor::Help);
			EZWIN32_CASE_TO_STR(Cursor::Arrow);
			EZWIN32_CASE_TO_STR(Cursor::Cross);
			EZWIN32_CASE_TO_STR(Cursor::IBeam);
			EZWIN32_CASE_TO_STR(Cursor::SizeWE);
			EZWIN32_CASE_TO_STR(Cursor::SizeNS);
			EZWIN32_CASE_TO_STR(Cursor::SizeAll);
			EZWIN32_CASE_TO_STR(Cursor::UpArrow);
			EZWIN32_CASE_TO_STR(Cursor::SizeNWSE);
			EZWIN32_CASE_TO_STR(Cursor::SizeNESW);
			EZWIN32_CASE_TO_STR(Cursor::AppStarting);
			default:	return "Cursor::Unknown";
		}
	}

	/*****************************************************************************
	****************************    HitTestResult    *****************************
	*****************************************************************************/

	//!	@brief	HitTest result type, returned by WM_NCHITTEST.
	enum class HitTestResult : Result
	{
		Nowhere			= HTNOWHERE,		//!< No part of the window
		Client			= HTCLIENT,			//!< Client area
		Caption			= HTCAPTION,		//!< Title bar (caption)
		SystemMenu		= HTSYSMENU,		//!< System menu (icon)
		GrowBox			= HTSIZE,			//!< Size box (legacy)
		Menu			= HTMENU,			//!< Menu area
		HScroll			= HTHSCROLL,		//!< Horizontal scroll bar
		VScroll			= HTVSCROLL,		//!< Vertical scroll bar
		MinButton		= HTMINBUTTON,		//!< Minimize button
		MaxButton		= HTMAXBUTTON,		//!< Maximize button
		Left			= HTLEFT,			//!< Left border
		Right			= HTRIGHT,			//!< Right border
		Top				= HTTOP,			//!< Top border
		TopLeft			= HTTOPLEFT,		//!< Top-left corner
		TopRight		= HTTOPRIGHT,		//!< Top-right corner
		Bottom			= HTBOTTOM,			//!< Bottom border
		BottomLeft		= HTBOTTOMLEFT,		//!< Bottom-left corner
		BottomRight		= HTBOTTOMRIGHT,	//!< Bottom-right corner
		Border			= HTBORDER,			//!< Non-sizing border
		CloseButton		= HTCLOSE,			//!< Close button
		HelpButton		= HTHELP,			//!< Help button
		Default			= -1,				//!< Use default hit test (DefWindowProc)
	};

	//!	@brief	Convert `HitTestResult` enum to string for debugging or logging.
	static inline const char* to_string(HitTestResult result)
	{
		switch (result)
		{
			EZWIN32_CASE_TO_STR(HitTestResult::Nowhere);
			EZWIN32_CASE_TO_STR(HitTestResult::Client);
			EZWIN32_CASE_TO_STR(HitTestResult::Caption);
			EZWIN32_CASE_TO_STR(HitTestResult::SystemMenu);
			EZWIN32_CASE_TO_STR(HitTestResult::GrowBox);
			EZWIN32_CASE_TO_STR(HitTestResult::Menu);
			EZWIN32_CASE_TO_STR(HitTestResult::HScroll);
			EZWIN32_CASE_TO_STR(HitTestResult::VScroll);
			EZWIN32_CASE_TO_STR(HitTestResult::MinButton);
			EZWIN32_CASE_TO_STR(HitTestResult::MaxButton);
			EZWIN32_CASE_TO_STR(HitTestResult::Left);
			EZWIN32_CASE_TO_STR(HitTestResult::Right);
			EZWIN32_CASE_TO_STR(HitTestResult::Top);
			EZWIN32_CASE_TO_STR(HitTestResult::TopLeft);
			EZWIN32_CASE_TO_STR(HitTestResult::TopRight);
			EZWIN32_CASE_TO_STR(HitTestResult::Bottom);
			EZWIN32_CASE_TO_STR(HitTestResult::BottomLeft);
			EZWIN32_CASE_TO_STR(HitTestResult::BottomRight);
			EZWIN32_CASE_TO_STR(HitTestResult::Border);
			EZWIN32_CASE_TO_STR(HitTestResult::CloseButton);
			EZWIN32_CASE_TO_STR(HitTestResult::HelpButton);
			EZWIN32_CASE_TO_STR(HitTestResult::Default);
			default:	return "HitTestResult::Unknown";
		}
	}

	/*****************************************************************************
	****************************    ThreadWindows    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Namespace for handling messages across all windows in the current thread.
	 *	@details	Provides utility functions for processing and waiting for messages
	 *				in the message queue of the current thread, affecting all associated windows.
	 */
	namespace ThreadWindows
	{
		/**
		 * @brief      Waits for and processes the next message for all windows belonging to the current thread.
		 * @details    This function blocks until a message is available in the message queue of the
		 *             current thread (regardless of which window it belongs to).
		 * @note       Unlike per-window message processing, passing `nullptr` as `hWnd` allows
		 *             messages for any window in this thread to be retrieved.
		 */
		void waitEvent();


		/**
		 * @brief      Processes all pending messages, if available.
		 * @details    Uses `PeekMessage` to check for all messages in the queue for this thread.
		 *             If a message exists, it is translated and dispatched to the window procedure.
		 * @return     `true` if a message was processed, `false` if no messages were pending.
		 */
		bool processEvents();
	}
}

/*********************************************************************************
********************************    Type alias    ********************************
*********************************************************************************/

using EzKey = easywin32::Key;
using EzByte = easywin32::Byte;
using EzSize = easywin32::Size;
using EzRect = easywin32::Rect;
using EzPoint = easywin32::Point;
using EzStyle = easywin32::Style;
using EzResult = easywin32::Result;
using EzWindow = easywin32::Window;
using EzCursor = easywin32::Cursor;
using EzExStyle = easywin32::ExStyle;
using EzBackdrop = easywin32::Backdrop;
using EzColorRGB = easywin32::ColorRGB;
using EzColorRGBA = easywin32::ColorRGBA;
using EzKeyAction = easywin32::KeyAction;
using EzMouseState = easywin32::MouseState;
using EzMouseAction = easywin32::MouseAction;
using EzMouseButton = easywin32::MouseButton;
using EzCornerStyle = easywin32::CornerStyle;
using EzHitTestResult = easywin32::HitTestResult;
using EzMouseStateFlags = easywin32::Flags<easywin32::MouseState>;
using EzNonClientRenderingPolicy = easywin32::NonClientRenderingPolicy;
template<typename EnumType> using EzFlags = easywin32::Flags<EnumType>;
namespace EzThreadWindows = easywin32::ThreadWindows;

/*********************************************************************************
**********************************    Window    **********************************
*********************************************************************************/

/**
 *	@brief		
 */
class easywin32::Window
{

public:

	Window() = default;

	Window(const Window&) = delete;

	void operator=(const Window&) = delete;

	Window(string_type title) { this->setTitle(title); }

	~Window() { this->close(); }

public:

	/**
	 *	@brief		Opens a new window with the specified title, bounds and styles.
	 *	@details	Creates a Win32 window using `CreateWindowEx`, applying the provided style flags and extended style flags.
	 *				Adjusts the window's bounding rectangle to account for non-client areas (e.g., borders, title bar) based on the style flags.
	 *				If the window is already open (m_hWnd is valid), this method does nothing. Stores the window handle and updates m_styleFlags and m_bounds.
	 *	@param[in]	title - The title of the window.
	 *	@param[in]	bounds - The rectangular area defining the window's position (left, top) and bounds (right, bottom) in screen coordinates.
	 *	@param[in]	styleFlags - The combination of window styles (e.g., `Style::OverlappedWindow`). Defaults to `Style::OverlappedWindow`.
	 *	@param[in]	exStyleFlags - The combination of extended window styles (e.g., ExStyle::Layered, ExStyle::TopMost). Defaults to 0 (no extended styles).
	 */
	void open(string_type title, Rect bounds, Flags<Style> styleFlags = Style::OverlappedWindow, Flags<ExStyle> exStyleFlags = 0);

	//!	@brief		Opens a new window with default title and specified bounds and styles.
	void open(Rect bounds, Flags<Style> styleFlags = Style::OverlappedWindow, Flags<ExStyle> exStyleFlags = 0)
	{
	#ifdef UNICODE
		this->open(L"Easy-Win32", bounds, styleFlags, exStyleFlags);
	#else
		this->open("Easy-Win32", bounds, styleFlags, exStyleFlags);
	#endif
	}


	/**
	 *	@brief		Opens a new window with the specified title, size and styles.
	 *	@details	Creates a Win32 window using `CreateWindowEx`, applying the provided style flags and extended style flags.
	 *				The window is positioned at default screen coordinates (CW_USEDEFAULT) and sized according to the provided Size structure.
	 *				Adjusts the window's bounding rectangle to account for non-client areas (e.g., borders, title bar) based on the style flags.
	 *				If the window is already open (m_hWnd is valid), this method does nothing. Updates m_styleFlags and m_bounds accordingly.
	 *	@param[in]	title - The title of the window.
	 *	@param[in]	with - The with of the window in pixels.
	 *	@param[in]	height - The height of the window in pixels.
	 *	@param[in]	styleFlags - The combination of window styles (e.g., `Style::OverlappedWindow`). Defaults to `Style::OverlappedWindow`.
	 *	@param[in]	exStyleFlags - The combination of extended window styles (e.g., `ExStyle::Layered`, `ExStyle::TopMost`). Defaults to 0 (no extended styles).
	 *	@note		The window is created at default screen coordinates (CW_USEDEFAULT). Use setPos to specify a custom position.
	 */
	void open(string_type title, int width, int height, Flags<Style> styleFlags = Style::OverlappedWindow, Flags<ExStyle> exStyleFlags = 0);

	//!	@brief		Opens a new window with default title and specified size and styles.
	void open(int width, int height, Flags<Style> styleFlags = Style::OverlappedWindow, Flags<ExStyle> exStyleFlags = 0)
	{
	#ifdef UNICODE
		this->open(L"Easy-Win32", width, height, styleFlags, exStyleFlags);
	#else
		this->open("Easy-Win32", width, height, styleFlags, exStyleFlags);
	#endif
	}


	/**
	 *	@brief		Closes and destroys the window.
	 *	@details	Calls DestroyWindow to close the window and releases the associated window handle (m_hWnd).
	 *				Sets m_hWnd to nullptr to indicate the window is no longer valid. Safe to call even if the window is already closed.
	 */
	void close() { ::DestroyWindow(m_hWnd);		m_hWnd = nullptr;	m_enableBlurBeind = false; }

public:

	//!	@brief	Gets style flags of the window.
	Flags<Style> getStyleFlags() const;

	//!	@brief	Returns the native handle of the win32 window.
	HWND nativeHandle() { return m_hWnd; }

	//!	@brief	Get the minimum window size allowed during resizing.
	Size getMinTrackSize() const { return m_minTrackSize; }

	//!	@brief	Get the maximum window size allowed during resizing.
	Size getMaxTrackSize() const { return m_maxTrackSize; }

	//!	@brief	Whether the specified window handle identifies an existing window (opened).	
	bool isOpen() const { return ::IsWindow(m_hWnd); }

	//!	@brief	Whether the window is minimized.
	bool isMinimized() const { return ::IsIconic(m_hWnd); }

	//!	@brief	Whether the window is maximized.
	bool isMaximized() const { return ::IsZoomed(m_hWnd); }

	//!	@brief	Whether the windows is focused.
	bool isFocused() const { return ::GetFocus() == m_hWnd; }

	//!	@brief	Checks if the window is currently marked as visible.
	bool isVisible() const { return ::IsWindowVisible(m_hWnd) != FALSE; }

	//!	@brief	Whether the windows is the foreground (active) window.
	bool isForeground() const { return ::GetForegroundWindow() == m_hWnd; }

	//!	@brief	Gets ex-style flags of the window.
	Flags<ExStyle> getExStyleFlags() const { return (DWORD)::GetWindowLongPtr(m_hWnd, GWL_EXSTYLE); }

	//!	@brief	Retrieves the position of the client area (left-top corner).
	Point getClientPos() const { Point pt = { 0, 0 };	::ClientToScreen(m_hWnd, &pt);		return pt; }

	//!	@brief	Gets the mouse position in client coordinates.
	Point getMousePos() const { Point pt = {};		::GetCursorPos(&pt);	::ScreenToClient(m_hWnd, &pt);		return pt; }

	//!	@brief	Returns the size of the client area (width and height) of the window in pixels.
	Size getClientExtent() const { Rect rect = {};	::GetClientRect(m_hWnd, &rect);		return Size{ rect.right, rect.bottom }; }

public:

	//!	@brief	Bring the window to front and set input focus.
	void setFocus() { ::SetFocus(m_hWnd); }

	//!	@brief	Rest style of the window.
	void setStyle(Flags<Style> styleFlags);

	//!	@brief	Rest ex-style of the window.
	void setExStyle(Flags<ExStyle> styleFlags);

	//!	@brief	Hide the window.
	void hide() { ::ShowWindow(m_hWnd, SW_HIDE); }

	//!	@brief	Bring the window to the top of the Z order.
	void bringToTop() { ::BringWindowToTop(m_hWnd); }

	//!	@brief	Show the window.
	void show() { ::ShowWindow(m_hWnd, SW_SHOWNORMAL); }

	//!	@brief	Displays or hides the cursor.
	void showCursor(bool bShow) { ::ShowCursor(bShow); }

	//!	@brief	Set position of the window.
	void setPos(int left, int top);
	void setPos(int left, int top, int right, int bottom);

	//!	@brief	Centers the window in the current monitor's work area.
	void centerToScreen();

	//!	@brief	Foreground the window.
	void setForeground() { ::SetForegroundWindow(m_hWnd); }

	//!	@brief	Destorys a timer.
	void killTimer(UINT_PTR id) { ::KillTimer(m_hWnd, id); }

	//!	@brief	Restore the winodw if it is minimized or maximized.
	void restore() { ::ShowWindow(m_hWnd, SW_RESTORE); }

	//!	@brief	Maximized the window.
	void maximize() { ::ShowWindow(m_hWnd, SW_SHOWMAXIMIZED); }

	//!	@brief	Minimized the window.
	void minimize() { ::ShowWindow(m_hWnd, SW_SHOWMINIMIZED); }

	//!	@brief	Set the minimum size the window can be resized to.
	void setMinTrackSize(Size minSize) { m_minTrackSize = minSize; }

	//!	@brief	Set the maximum size the window can be resized to.
	void setMaxTrackSize(Size maxSize) { m_maxTrackSize = maxSize; }

	//!	@brief	Enables or disables mouse and keyboard input to the specified window.
	void enableInput(bool bEnable) { ::EnableWindow(m_hWnd, bEnable); }

	//!	@brief	Sets the window title text.	
	void setTitle(string_type title) { ::SetWindowText(m_hWnd, title.c_str()); }

	//!	@brief	Sets the parent window
	void setParent(Window & parent) { ::SetParent(m_hWnd, parent.nativeHandle()); }

	//!	@brief	Converts the screen coordinates to client coordinates.
	Point screenToClient(Point pt) const { ::ScreenToClient(m_hWnd, &pt);	return pt; }

	//!	@brief	Converts the client coordinates to screen coordinates.
	Point clientToScreen(Point pt) const { ::ClientToScreen(m_hWnd, &pt);	return pt; }

	//!	@brief	Draws a raw RGB bitmap onto the window at the specified position.
	void drawBitmap(const ColorRGB * pixels, int width, int height, int dstX = 0, int dstY = 0);

	//!	@brief	Creates a timer with the specified id and time-out value.
	void setTimer(UINT_PTR id, unsigned int millisecond) { ::SetTimer(m_hWnd, id, millisecond, NULL); }

	//!	@brief	Sets the window to stay on top of others.
	void setAlwaysOnTop(bool enable) { ::SetWindowPos(m_hWnd, enable ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE); }

	/**
	 *	@brief		Simulates a non-client area mouse click (e.g., title bar or window border) at the current cursor position.
	 *	@details	This function sends a WM_NCLBUTTONDOWN message to the window, making the system believe
	 *				that the user has clicked on the specified non-client area. For example, passing HTCAPTION
	 *				will cause the window to start moving as if the title bar was clicked.
	 */
	void triggerNonClientClick(HitTestResult result)
	{
		POINT pt = {};		::GetCursorPos(&pt);
		
		::SendMessage(m_hWnd, WM_NCLBUTTONDOWN, static_cast<WPARAM>(result), MAKELPARAM(pt.x, pt.y));
	}

	//!	@brief	Requests the window to be redrawn.
	void requestRedraw(const Rect * rect = nullptr, bool eraseBackground = false)
	{
		::InvalidateRect(m_hWnd, rect, eraseBackground);
		::UpdateWindow(m_hWnd);
	}

	//!	@brief	Specify the cursor shape, must be called from the main thread.
	void setCursor(Cursor cursor)
	{
	#ifdef UNICODE
		::SetCursor(::LoadCursor(NULL, (LPCWSTR)cursor));
	#else
		::SetCursor(::LoadCursor(NULL, (LPCSTR)cursor));
	#endif
	}

public:	// Layered window section

	//!	@brief	Gets the window opacity.
	Byte getOpacity() const { return m_opacity; }

	//!	@brief	Sets the window opacity (0-255, requires ExStyle::Layered).
	void setOpacity(Byte opacity) { ::SetLayeredWindowAttributes(m_hWnd, 0, m_opacity = opacity, LWA_ALPHA); }

	//!	@brief		Sets a color key for the layered window to enable per-pixel transparency (requires ExStyle::Layered).
	//! @details	All pixels exactly matching the given color will be rendered as transparent, allowing content behind the window to show through.
	void setColorKey(ColorRGB color) { ::SetLayeredWindowAttributes(m_hWnd, RGB(color.r, color.g, color.b), 0, LWA_COLORKEY); }

public: // DWM section (get)

	//!	@brief	Get non-client rendering policy.
	NonClientRenderingPolicy getNonClientRenderingPolicy() const { auto val = NonClientRenderingPolicy::UseWindowStyle;	::DwmGetWindowAttribute(m_hWnd, DWMWA_NCRENDERING_POLICY, &val, sizeof(val));	return val; }

	//!	@brief	Set color of caption/text/border (requires Windows11)
	ColorRGB getCaptionColor() const { COLORREF val = RGB(0, 0, 0);	::DwmGetWindowAttribute(m_hWnd, DWMWA_CAPTION_COLOR, &val, sizeof(val));	return ColorRGB{ GetRValue(val), GetGValue(val), GetBValue(val) }; }
	ColorRGB getBorderColor() const { COLORREF val = RGB(0, 0, 0);	::DwmGetWindowAttribute(m_hWnd, DWMWA_BORDER_COLOR, &val, sizeof(val));		return ColorRGB{ GetRValue(val), GetGValue(val), GetBValue(val) }; }
	ColorRGB getTextColor() const { COLORREF val = RGB(0, 0, 0);	::DwmGetWindowAttribute(m_hWnd, DWMWA_TEXT_COLOR, &val, sizeof(val));		return ColorRGB{ GetRValue(val), GetGValue(val), GetBValue(val) }; }
	
	//!	@brief	Get the corner style of the window.
	CornerStyle getCornerStyle() const { auto val = CornerStyle::Default;	::DwmGetWindowAttribute(m_hWnd, DWMWA_WINDOW_CORNER_PREFERENCE, &val, sizeof(val));  return val; }

	//!	@brief	Get the backdrop of the window
	Backdrop getBackdrop() const { auto val = Backdrop::Auto;	::DwmGetWindowAttribute(m_hWnd, DWMWA_SYSTEMBACKDROP_TYPE, &val, sizeof(val));	 return val; }
	
	//!	@brief	Checks if the dark mode is enabled.
	bool immersiveDarkModeEnabled() const { BOOL val = FALSE;	::DwmGetWindowAttribute(m_hWnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &val, sizeof(val));	 return val; }

	//!	@brief	Checks if the DWM "blur-behind" effect is enabled.
	bool blurBeindWindowEnabled() const { return m_enableBlurBeind; }
	
public: // DWM section (set)

	//! @brief	Allows the window to either use the accent color, or dark, according to the user ColorRGB Mode preferences.
	void enableImmersiveDarkMode(bool enable) { BOOL val = enable;	::DwmSetWindowAttribute(m_hWnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &val, sizeof(val)); }

	//!	@brief	Set non-client rendering policy.
	void setNonClientRenderingPolicy(NonClientRenderingPolicy policy) { ::DwmSetWindowAttribute(m_hWnd, DWMWA_NCRENDERING_POLICY, &policy, sizeof(policy)); }
	
	//!	@brief	Extends the window frame into the client area.
	void extendFrameIntoClientArea(int left, int top, int right, int bottom) { MARGINS margins = { left, right, top, bottom };	::DwmExtendFrameIntoClientArea(m_hWnd, &margins); }

	//!	@brief	Set color of caption/text/border (requires Windows11)
	void setCaptionColor(ColorRGB color) { COLORREF val = RGB(color.r, color.g, color.b);	::DwmSetWindowAttribute(m_hWnd, DWMWA_CAPTION_COLOR, &val, sizeof(val)); }
	void setBorderColor(ColorRGB color) { COLORREF val = RGB(color.r, color.g, color.b);	::DwmSetWindowAttribute(m_hWnd, DWMWA_BORDER_COLOR, &val, sizeof(val)); }
	void setTextColor(ColorRGB color) { COLORREF val = RGB(color.r, color.g, color.b);		::DwmSetWindowAttribute(m_hWnd, DWMWA_TEXT_COLOR, &val, sizeof(val)); }

	//!	@brief	Set corner style of the window (requires Windows11)
	void setCornerStyle(CornerStyle style) { ::DwmSetWindowAttribute(m_hWnd, DWMWA_WINDOW_CORNER_PREFERENCE, &style, sizeof(style)); }

	//!	@brief	Controls the system-drawn backdrop material of a window, including behind the non-client area.
	void setBackdrop(Backdrop backdrop){ ::DwmSetWindowAttribute(m_hWnd, DWMWA_SYSTEMBACKDROP_TYPE, &backdrop, sizeof(backdrop)); }

	//!	@brief	Enables or disables the DWM "blur-behind" effect for the window (aka. alpha-composition).
	void enableBlurBeindWindow(bool enable);

public:

	/**
	 *	@brief		Waits for and processes the next window message.
	 *	@details	This call will block until a new message is available in the queue for this window.
	 *				It retrieves the message with `GetMessage`, translates it (e.g. converts virtual-key
	 *				messages into character messages), and dispatches it to the window procedure.
	 */
	void waitEvent();


	/**
	 *	@brief		Processes all pending window messages, if available.
	 *	@details	Uses `PeekMessage` to check for all messages in the queue for this window.
	 *				If a message exists, it is translated and dispatched to the window procedure.
	 *	@return		`true` if a message was processed, `false` if no messages were pending.
	 */
	bool processEvents();

private:

	//!	@brief	Adjusts the window rectangle based on the specified styles.
	template<bool SkipCaption> static void adjustWindowRect(Rect & rect, DWORD dwStyle, DWORD dwExStyle);

	//!	@brief	Window procedure for message dispatching.
	template<bool SkipCaption> static LRESULT procedure(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam);

	//!	@brief	Create window (internal).
	template<bool SkipCaption> void internalOpen(string_type title, DWORD dwStyle, DWORD dwExStyle, int x, int y, int w, int h);

	//!	@brief	Determines whether the window should skip the caption area based on style flags.
	static bool shouldSkipCaption(Flags<Style> styleFlags) { return !styleFlags.has(Style::Caption) && styleFlags.has(Style::ThickFrame); }

	//!	@brief	Converts internal style flags to native Win32 style bits.
	static DWORD toNativeStyle(Flags<Style> styleFlags);

public:

	/**
	 *	@brief		User-defined callback functions, called when triggered and defined by the user.
	 *	@note		Each callback can optionally handle the event and determine whether it should be passed to the 
	 *				system's default handler (`DefWindowProc`).
	 *	@details	If the callback returns any value other than `-1`, that value will be returned directly as the 
	 *				window procedure’s result — meaning the event is considered handled and `DefWindowProc` will **not** be called.
	 *	@details	If the callback returns `-1`, the message will be forwarded to `DefWindowProc` for default processing.
	 */
	std::function<Result(HWND, UINT, WPARAM, LPARAM)>						forwardMessage;		// Called before default message handling.
	std::function<Result(wchar_t)>											onInputCharacter;	// Called when a character input (WM_CHAR, WM_SYSCHAR, WM_UNICHAR) is received.
	std::function<Result(const std::vector<string_type>&)>					onDropFiles;		// Called when files are dropped onto the window (WM_DROPFILES), requires ExStyle::AcceptFiles.
	std::function<Result()>													onEnterMove;		// Called when the user starts moving or resizing the window(WM_ENTERSIZEMOVE).
	std::function<Result()>													onExitMove;			// Called when the user finishes moving or resizing the window (WM_EXITSIZEMOVE).
	std::function<Result()>													onPaint;			// Called when the window needs to be repainted (WM_PAINT).
	std::function<Result(UINT_PTR id)>										onTimer;			// Called when a timer event occurs (WM_TIMER).
	std::function<Result(bool focused)>										onFocus;			// Called when the window gains or lost focus (WM_SETFOCUS, WM_KILLFOCUS).
	std::function<Result()>													onClose;			// Called when the window is about to close (WM_CLOSE).
	std::function<Result()>													onMouseLeave;		// Called when the mouse leave the client area (WM_MOUSELEAVE).
	std::function<Result(int x, int y)>										onMove;				// Called when the window is moved (WM_MOVE).
	std::function<Result(int w, int h)>										onResize;			// Called when the window is resized (WM_SIZE).
	std::function<Result(int x, int y, Flags<MouseState>)>					onMouseMove;		// Called when the mouse is moved (WM_MOUSEMOVE).
	std::function<Result(int dx, int dy, Flags<MouseState>)>				onWheelScroll;		// Called when the mouse wheel is scrolled (WM_MOUSEWHEEL / WM_MOUSEHWHEEL).
	std::function<Result(MouseButton, MouseAction, Flags<MouseState>)>		onMouseClick;		// Called when a mouse button event occurs (down, up, or double click).
	std::function<Result(Key, KeyAction)>									onKeyboardPress;	// Called when a key is pressed, released, or repeated (WM_KEYDOWN/WM_KEYUP).
	std::function<HitTestResult(int x, int y)>								onHitTest;			// Called when WM_NCHITTEST is received, should return the hit-test result based on cursor position.

private:

	HWND	m_hWnd = nullptr;
	Size	m_minTrackSize = { 120, 31 };
	Size	m_maxTrackSize = { LONG_MAX, LONG_MAX };
	bool	m_enableBlurBeind = false;
	bool	m_skipCaption = false;
	Byte	m_opacity = 255;	// 0 = transparent, 255 = opaque
};

/*********************************************************************************
******************************    Implementation    ******************************
*********************************************************************************/

#ifdef EZWIN32_IMPLEMENTATION

#include <windowsx.h>

/**
 *	@brief		Static window procedure for message dispatching.
 *	@details	This function is registered with the Win32 API as the window procedure
 *				for the Window class. It handles core messages (e.g., `WM_CREATE`, `WM_DESTROY`),
 *				associates HWND handles with Window instances, and forwards other messages to user-defined callbacks stored in the Window object.
 *	@param[in]	hWnd - Handle to the window receiving the message.
 *	@param[in]	uMsg - The message identifier (e.g., `WM_CREATE`, `WM_SIZE`, `WM_KEYDOWN`).
 *	@param[in]	wParam - Additional message-specific information.
 *	@param[in]	lParam - Additional message-specific information.
 *	@return		The result of message processing. If the message is not handled, the result of `DefWindowProc()` is returned.
 *	@note
 *		- On `WM_CREATE`, the Window pointer passed via lpCreateParams is stored in the window's user data slot for later retrieval.
 *		- On `WM_DESTROY`, a quit message is posted to end the application loop.
 *		- For other messages, the stored Window pointer is used to dispatch events to registered callbacks (e.g., onClose, onResize, onMouseClick).
 */
template<bool SkipCaption> LRESULT easywin32::Window::procedure(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	// Retrieve the stored Window* from the window's user data
	auto window = reinterpret_cast<Window*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));

	if (uMsg == WM_CREATE)	// Retrieve the use pointer
	{
		CREATESTRUCT * createStrcut = reinterpret_cast<CREATESTRUCT*>(lParam);

		::SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(createStrcut->lpCreateParams));
	}
	else if (uMsg == WM_NCCALCSIZE)
	{
		if constexpr (SkipCaption)
		{
			// When skipping the caption area, adjust the client region manually.
			NCCALCSIZE_PARAMS * params = (NCCALCSIZE_PARAMS*)lParam;

			long top = params->rgrc[0].top;		// Backup original top value

			// Let the default window proc calculate the base frame
			LRESULT res = ::DefWindowProc(hWnd, uMsg, wParam, lParam);

			if (::IsZoomed(hWnd))
			{
				constexpr int frameSize = 8;		//	win32 default size for drag

				// When maximized, keep a small draggable frame area at the top
				params->rgrc[0].top = top + frameSize;
			}
			else
			{
				// Otherwise, keep the top edge unchanged to fully skip caption area
				params->rgrc[0].top = top;
			}

			return res;
		}
	}
	else if (uMsg == WM_GETMINMAXINFO)
	{
		if (window != nullptr)
		{
			MINMAXINFO * mmi = (MINMAXINFO*)lParam;

			mmi->ptMinTrackSize.x = window->m_minTrackSize.cx;
			mmi->ptMinTrackSize.y = window->m_minTrackSize.cy;
			mmi->ptMaxTrackSize.x = window->m_maxTrackSize.cx;
			mmi->ptMaxTrackSize.y = window->m_maxTrackSize.cy;

			return 0;
		}
	}
	else if (uMsg == WM_DESTROY)	// Handle window destruction
	{
		::PostQuitMessage(0);		// Post a quit message to end the application message loop
		
		return 0;	// Message handled
	}
	else if (window != nullptr)
	{
		Result result = -1;

		// Forward messages to the user-defined callback first.
		if (window->forwardMessage)
		{
			result = window->forwardMessage(hWnd, uMsg, wParam, lParam);

			if (result != -1)
				return result;
		}

		//	Dispatch standard messages to callbacks
		switch (uMsg)
		{
			case WM_CLOSE:			if (window->onClose)			result = window->onClose();			break;
			case WM_PAINT:			if (window->onPaint)			result = window->onPaint();			break;
			case WM_TIMER:			if (window->onTimer)			result = window->onTimer(wParam);	break;
			case WM_SETFOCUS:		if (window->onFocus)			result = window->onFocus(true);		break;
			case WM_KILLFOCUS:		if (window->onFocus)			result = window->onFocus(false);	break;
			case WM_MOUSELEAVE:		if (window->onMouseLeave)		result = window->onMouseLeave();	break;
			case WM_EXITSIZEMOVE:	if (window->onExitMove)			result = window->onExitMove();		break;
			case WM_ENTERSIZEMOVE:	if (window->onEnterMove)		result = window->onEnterMove();		break;

			case WM_CHAR:			if (window->onInputCharacter)	result = window->onInputCharacter(static_cast<wchar_t>(wParam));	break;
			case WM_SYSCHAR:		if (window->onInputCharacter)	result = window->onInputCharacter(static_cast<wchar_t>(wParam));	break;
			case WM_UNICHAR:		if (window->onInputCharacter)	result = window->onInputCharacter(static_cast<wchar_t>(wParam));	break;

			case WM_MOVE:			if (window->onMove)				result = window->onMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));				break;
			case WM_SIZE:			if (window->onResize)			result = window->onResize(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));				break;
			case WM_NCHITTEST:		if (window->onHitTest)			result = (Result)window->onHitTest(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));		break;

			case WM_KEYUP:			if (window->onKeyboardPress)	result = window->onKeyboardPress(static_cast<Key>(wParam), KeyAction::Release);		break;
			case WM_SYSKEYUP:		if (window->onKeyboardPress)	result = window->onKeyboardPress(static_cast<Key>(wParam), KeyAction::Release);		break;

			case WM_KEYDOWN:		if (window->onKeyboardPress)	result = window->onKeyboardPress(static_cast<Key>(wParam), (lParam & (1 << 30)) ? KeyAction::Repeat : KeyAction::Press);		break;
			case WM_SYSKEYDOWN:		if (window->onKeyboardPress)	result = window->onKeyboardPress(static_cast<Key>(wParam), (lParam & (1 << 30)) ? KeyAction::Repeat : KeyAction::Press);		break;

			case WM_MOUSEMOVE:		if (window->onMouseMove)		result = window->onMouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), static_cast<MouseState>(wParam & 127));				break;
			case WM_MOUSEWHEEL:		if (window->onWheelScroll)		result = window->onWheelScroll(0, static_cast<SHORT>(HIWORD(wParam)) / WHEEL_DELTA, static_cast<MouseState>(wParam & 127));		break;
			case WM_MOUSEHWHEEL:	if (window->onWheelScroll)		result = window->onWheelScroll(static_cast<SHORT>(HIWORD(wParam)) / WHEEL_DELTA, 0, static_cast<MouseState>(wParam & 127));		break;

			case WM_LBUTTONUP:		if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Left, MouseAction::Up, static_cast<MouseState>(wParam & 127));		break;
			case WM_RBUTTONUP:		if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Right, MouseAction::Up, static_cast<MouseState>(wParam & 127));		break;
			case WM_MBUTTONUP:		if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Middle, MouseAction::Up, static_cast<MouseState>(wParam & 127));		break;

			case WM_LBUTTONDOWN:	if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Left, MouseAction::Down, static_cast<MouseState>(wParam & 127));			break;
			case WM_RBUTTONDOWN:	if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Right, MouseAction::Down, static_cast<MouseState>(wParam & 127));		break;
			case WM_MBUTTONDOWN:	if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Middle, MouseAction::Down, static_cast<MouseState>(wParam & 127));		break;

			case WM_LBUTTONDBLCLK:	if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Left, MouseAction::DoubleClick, static_cast<MouseState>(wParam & 127));		break;
			case WM_RBUTTONDBLCLK:	if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Right, MouseAction::DoubleClick, static_cast<MouseState>(wParam & 127));		break;
			case WM_MBUTTONDBLCLK:	if (window->onMouseClick)		result = window->onMouseClick(MouseButton::Middle, MouseAction::DoubleClick, static_cast<MouseState>(wParam & 127));	break;

			case WM_XBUTTONUP:
			case WM_XBUTTONDOWN:
			case WM_XBUTTONDBLCLK:
			{
				if (window->onMouseClick)
				{
					auto button = GET_XBUTTON_WPARAM(wParam);

					auto action = (uMsg == WM_XBUTTONUP) ? MouseAction::Up : (uMsg == WM_XBUTTONDOWN) ? MouseAction::Down : MouseAction::DoubleClick;

					if (button == XBUTTON1)
					{
						result = window->onMouseClick(MouseButton::XButton1, action, static_cast<MouseState>(wParam & 127));
					}
					else if (button == XBUTTON2)
					{
						result = window->onMouseClick(MouseButton::XButton2, action, static_cast<MouseState>(wParam & 127));
					}
				}

				break;
			}
			case WM_DROPFILES:
			{
				HDROP hDrop = (HDROP)wParam;

				if (window->onDropFiles)
				{
					UINT count = ::DragQueryFile(hDrop, UINT32_MAX, nullptr, 0);

					std::vector<string_type> filePaths(count);

					for (UINT i = 0; i < count; ++i)
					{
						UINT bufferSize = ::DragQueryFile(hDrop, i, nullptr, 0);

						filePaths[i].resize(bufferSize + 1, '\0');

						::DragQueryFile(hDrop, i, filePaths[i].data(), bufferSize + 1);
					}

					result = window->onDropFiles(filePaths);
				}

				::DragFinish(hDrop);

				break;
			}
		}

		if (result != -1)
			return result;
	}

	//	Default handling for unprocessed messages
	return ::DefWindowProc(hWnd, uMsg, wParam, lParam);
}


/**
 *	@brief		Internal method to create a Win32 window with specified parameters.
 *	@details	Creates a window using `CreateWindowEx` with the provided title, styles, and dimensions.
 *				Registers a window class (either "EasyWin32" or "EasyWin32-NoTitleBar" based on EraseTitleBar)
 *				to handle message dispatching. The window class is registered once per application lifetime and
 *				unregistered upon destruction. Stores the window handle in m_hWnd and associates this Window instance
 *				with the native handle for message processing.
 *	@tparam		SkipCaption - If true, registers a window class without a title bar.
 *	@param[in]	title - The window title text, displayed in the title bar (if applicable).
 *	@param[in]	dwStyle - The native window style flags (e.g., WS_OVERLAPPEDWINDOW, WS_POPUP).
 *	@param[in]	dwExStyle - The native extended window style flags (e.g., WS_EX_LAYERED, WS_EX_TOPMOST).
 *	@param[in]	x - The x-coordinate of the window's top-left corner in screen coordinates (e.g., CW_USEDEFAULT).
 *	@param[in]	y - The y-coordinate of the window's top-left corner in screen coordinates (e.g., CW_USEDEFAULT).
 *	@param[in]	w - The width of the window in pixels, including non-client areas.
 *	@param[in]	h - The height of the window in pixels, including non-client areas.
 *	@note		The window class is registered only once per application, ensuring efficient resource usage.
 *	@note		If the window class registration fails, an assertion is triggered.
 */
template<bool SkipCaption> void easywin32::Window::internalOpen(string_type title, DWORD dwStyle, DWORD dwExStyle, int x, int y, int w, int h)
{
	/**
	 *	@brief	Static window class wrapper for Win32 window registration.
	 *
	 *	This struct inherits from `WNDCLASSEX` and is used to register a single
	 *	window class with the Win32 API. The class is automatically registered
	 *	upon construction and unregistered upon destruction. This ensures that
	 *	the registration occurs only once during the application's lifetime.
	 */
	static struct NativeClass : public WNDCLASSEX
	{
		bool isRegistered;

		NativeClass() : isRegistered(false)
		{
			cbSize			= sizeof(WNDCLASSEXW);
			style			= CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
			lpfnWndProc		= Window::procedure<SkipCaption>;
			cbClsExtra		= 0;
			cbWndExtra		= 0;
			hInstance		= ::GetModuleHandle(NULL);
			hIcon			= ::LoadIcon(NULL, IDI_APPLICATION);
			hCursor			= NULL;
			hbrBackground	= (HBRUSH)::GetStockObject(NULL_BRUSH);
			lpszMenuName	= NULL;
			hIconSm			= ::LoadIcon(NULL, IDI_WINLOGO);
		#ifdef UNICODE
			lpszClassName	= SkipCaption ? L"EasyWin32-SkipCaption" : L"EasyWin32";
		#else
			lpszClassName	= SkipCaption ? "EasyWin32-SkipCaption" : "EasyWin32";
		#endif
			isRegistered	= ::RegisterClassEx(this);
		}

		~NativeClass()
		{
			if (isRegistered)
			{
				::UnregisterClass(lpszClassName, hInstance);
			}
		}
	} s_win32Class;

	assert(s_win32Class.isRegistered);

	if (s_win32Class.isRegistered)
	{
		m_hWnd = ::CreateWindowEx(dwExStyle, s_win32Class.lpszClassName, title.c_str(), dwStyle,
								  x, y, w, h, NULL, NULL, s_win32Class.hInstance,
								  this /* Additional application data */);

		//! Keep in sync with the `m_opacity`.
		if (dwExStyle | WS_EX_LAYERED)
		{
			this->setOpacity(m_opacity);
		}
	}
}


/**
 *	@brief		Opens a new window with the specified title, bounds and styles.
 *	@details	Creates a Win32 window using `CreateWindowEx`, applying the provided style flags and extended style flags.
 *				Adjusts the window's bounding rectangle to account for non-client areas (e.g., borders, title bar) based on the style flags.
 *				If the window is already open (m_hWnd is valid), this method does nothing. Stores the window handle and updates m_styleFlags and m_bounds.
 *	@param[in]	title - The title of the window.
 *	@param[in]	bounds - The rectangular area defining the window's position (left, top) and bounds (right, bottom) in screen coordinates.
 *	@param[in]	styleFlags - The combination of window styles (e.g., `Style::OverlappedWindow`). Defaults to `Style::OverlappedWindow`.
 *	@param[in]	exStyleFlags - The combination of extended window styles (e.g., ExStyle::Layered, ExStyle::TopMost). Defaults to 0 (no extended styles).
 */
void easywin32::Window::open(string_type title, Rect bounds, Flags<Style> styleFlags, Flags<ExStyle> exStyleFlags)
{
	if (::IsWindow(m_hWnd) != 0)	return;

	DWORD dwStyle = Window::toNativeStyle(styleFlags);

	if (Window::shouldSkipCaption(styleFlags))
	{
		m_skipCaption = true;

		Window::adjustWindowRect<true>(bounds, dwStyle, exStyleFlags);

		this->internalOpen<true>(title, dwStyle, exStyleFlags, bounds.left, bounds.top,
								 bounds.right - bounds.left, bounds.bottom - bounds.top);
	}
	else
	{
		m_skipCaption = false;

		Window::adjustWindowRect<false>(bounds, dwStyle, exStyleFlags);

		this->internalOpen<false>(title, dwStyle, exStyleFlags, bounds.left, bounds.top,
								  bounds.right - bounds.left, bounds.bottom - bounds.top);
	}
}


/**
 *	@brief		Opens a new window with the specified title, size and styles.
 *	@details	Creates a Win32 window using `CreateWindowEx`, applying the provided style flags and extended style flags.
 *				The window is positioned at default screen coordinates (CW_USEDEFAULT) and sized according to the provided Size structure.
 *				Adjusts the window's bounding rectangle to account for non-client areas (e.g., borders, title bar) based on the style flags.
 *				If the window is already open (m_hWnd is valid), this method does nothing. Updates m_styleFlags and m_bounds accordingly.
 *	@param[in]	title - The title of the window.
 *	@param[in]	with - The with of the window in pixels.
 *	@param[in]	height - The height of the window in pixels.
 *	@param[in]	styleFlags - The combination of window styles (e.g., `Style::OverlappedWindow`). Defaults to `Style::OverlappedWindow`.
 *	@param[in]	exStyleFlags - The combination of extended window styles (e.g., `ExStyle::Layered`, `ExStyle::TopMost`). Defaults to 0 (no extended styles).
 *	@note		The window is created at default screen coordinates (CW_USEDEFAULT). Use setPos to specify a custom position.
 */
void easywin32::Window::open(string_type title, int width, int height, Flags<Style> styleFlags, Flags<ExStyle> exStyleFlags)
{
	if (::IsWindow(m_hWnd) != 0)	return;

	RECT bounds = { 0, 0, width, height };

	DWORD dwStyle = Window::toNativeStyle(styleFlags);

	if (Window::shouldSkipCaption(styleFlags))
	{
		m_skipCaption = true;

		Window::adjustWindowRect<true>(bounds, dwStyle, exStyleFlags);

		this->internalOpen<true>(title, dwStyle, exStyleFlags, CW_USEDEFAULT, CW_USEDEFAULT,
								 bounds.right - bounds.left, bounds.bottom - bounds.top);
	}
	else
	{
		m_skipCaption = false;

		Window::adjustWindowRect<false>(bounds, dwStyle, exStyleFlags);

		this->internalOpen<false>(title, dwStyle, exStyleFlags, CW_USEDEFAULT, CW_USEDEFAULT,
								  bounds.right - bounds.left, bounds.bottom - bounds.top);
	}
}


//! @brief	Adjust the given client rectangle based on window styles and extended styles.
template<bool SkipCaption> void easywin32::Window::adjustWindowRect(Rect & rect, DWORD dwStyle, DWORD dwExStyle)
{
	auto top = rect.top;

	::AdjustWindowRectEx(&rect, dwStyle, FALSE, dwExStyle);

	if constexpr (SkipCaption)
	{
		rect.top = top;		// Preserve top edge to skip caption adjustment
	}
}


//! @brief	Convert easywin32 style flags to native Win32 window style bits.
DWORD easywin32::Window::toNativeStyle(Flags<Style> styleFlags)
{
	DWORD dwStyle = WS_POPUP;	// Base style (Win32 auto-adds WS_CAPTION if not using WS_POPUP)
	
	if (styleFlags.has(Style::Border))
		dwStyle |= WS_BORDER;

	if (styleFlags.has(Style::Visible))
		dwStyle |= WS_VISIBLE;

	if (styleFlags.has(Style::SysMenu))
		dwStyle |= WS_SYSMENU;

	if (styleFlags.has(Style::Disabled))
		dwStyle |= WS_DISABLED;

	if (styleFlags.has(Style::Resizable))
		dwStyle |= WS_THICKFRAME;

	if (styleFlags.has(Style::MinimizeBox))
		dwStyle |= WS_MINIMIZEBOX;

	if (styleFlags.has(Style::MaximizeBox))
		dwStyle |= WS_MAXIMIZEBOX;

	// Add caption if explicitly requested or when thick frame is used (required by Win32 for proper window behavior)
	if (styleFlags.has(Style::Caption) || styleFlags.has(Style::ThickFrame))
		dwStyle |= WS_CAPTION;

	return dwStyle;
}


//!	@brief	Gets styles of the window.
easywin32::Flags<easywin32::Style> easywin32::Window::getStyleFlags() const
{
	DWORD dwStyle = (DWORD)::GetWindowLongPtr(m_hWnd, GWL_STYLE);

	Flags<Style> style = 0;
	
	if ((dwStyle & WS_BORDER) == WS_BORDER)
		style |= Style::Border;

	if ((dwStyle & WS_VISIBLE) == WS_VISIBLE)
		style |= Style::Visible;

	if ((dwStyle & WS_SYSMENU) == WS_SYSMENU)
		style |= Style::SysMenu;

	if ((dwStyle & WS_DISABLED) == WS_DISABLED)
		style |= Style::Disabled;

	if ((dwStyle & WS_THICKFRAME) == WS_THICKFRAME)
		style |= Style::Resizable;

	if ((dwStyle & WS_MINIMIZEBOX) == WS_MINIMIZEBOX)
		style |= Style::MinimizeBox;

	if ((dwStyle & WS_MAXIMIZEBOX) == WS_MAXIMIZEBOX)
		style |= Style::MaximizeBox;

	if ((dwStyle & WS_CAPTION) == WS_CAPTION)
	{
		style |= Style::ThickFrame;

		if (!m_skipCaption)
			style |= Style::Caption;
	}

	return style;
}


//!	@brief	Rest style of the window.
void easywin32::Window::setStyle(Flags<Style> styleFlags)
{
	auto pos = this->getClientPos();

	auto extent = this->getClientExtent();

	DWORD dwStyle = Window::toNativeStyle(styleFlags);

	Rect rect = { pos.x, pos.y, pos.x + extent.cx, pos.y + extent.cy };

	::SetWindowLongPtr(m_hWnd, GWL_STYLE, static_cast<LONG_PTR>(dwStyle));

	if (Window::shouldSkipCaption(styleFlags))
	{
		m_skipCaption = true;

		Window::adjustWindowRect<true>(rect, dwStyle, (DWORD)GetWindowLongPtr(m_hWnd, GWL_EXSTYLE));

		::SetWindowLongPtr(m_hWnd, GWLP_WNDPROC, (LONG_PTR)Window::procedure<true>);
	}
	else
	{
		m_skipCaption = false;

		Window::adjustWindowRect<false>(rect, dwStyle, (DWORD)GetWindowLongPtr(m_hWnd, GWL_EXSTYLE));

		::SetWindowLongPtr(m_hWnd, GWLP_WNDPROC, (LONG_PTR)Window::procedure<false>);
	}

	::SetWindowPos(m_hWnd, NULL, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, SWP_FRAMECHANGED | SWP_NOZORDER);
}


//!	@brief	Rest ex-style of the window.
void easywin32::Window::setExStyle(Flags<ExStyle> styleFlags)
{
	::SetWindowLongPtr(m_hWnd, GWL_EXSTYLE, static_cast<LONG_PTR>(styleFlags.mask));

	::SetWindowPos(m_hWnd, NULL, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE);

	//! Keep in sync with the `m_opacity`.
	if (styleFlags.has(ExStyle::Layered))
	{
		this->setOpacity(m_opacity);
	}
}


//!	@brief	Set position of the window.
void easywin32::Window::setPos(int left, int top)
{
	auto pos = this->getClientPos();

	if ((pos.x != left) || (pos.y != top))
	{
		Rect rect = { left, top, 0, 0 };	// Ignores the right and bottom parameters

		if (m_skipCaption)
			Window::adjustWindowRect<true>(rect, (DWORD)GetWindowLongPtr(m_hWnd, GWL_STYLE), (DWORD)GetWindowLongPtr(m_hWnd, GWL_EXSTYLE));
		else
			Window::adjustWindowRect<false>(rect, (DWORD)GetWindowLongPtr(m_hWnd, GWL_STYLE), (DWORD)GetWindowLongPtr(m_hWnd, GWL_EXSTYLE));

		::SetWindowPos(m_hWnd, NULL, rect.left, rect.top, 0, 0, SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOSIZE | SWP_NOREDRAW);
	}
}


//!	@brief	Set position of the window.
void easywin32::Window::setPos(int left, int top, int right, int bottom)
{
	auto extent = this->getClientExtent();

	if ((right - left == extent.cx) && (bottom - top == extent.cy))
	{
		this->setPos(left, top);
	}
	else
	{
		Rect rect = { left, top, right, bottom };

		if (m_skipCaption)
			Window::adjustWindowRect<true>(rect, (DWORD)GetWindowLongPtr(m_hWnd, GWL_STYLE), (DWORD)GetWindowLongPtr(m_hWnd, GWL_EXSTYLE));
		else
			Window::adjustWindowRect<false>(rect, (DWORD)GetWindowLongPtr(m_hWnd, GWL_STYLE), (DWORD)GetWindowLongPtr(m_hWnd, GWL_EXSTYLE));

		::SetWindowPos(m_hWnd, NULL, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, SWP_FRAMECHANGED | SWP_NOZORDER);
	}
}


//!	@brief	Centers the window in the current monitor's work area.
void easywin32::Window::centerToScreen()
{
	HMONITOR hMon = ::MonitorFromWindow(m_hWnd, MONITOR_DEFAULTTONEAREST);
	MONITORINFO mi = { sizeof(mi) };
	::GetMonitorInfo(hMon, &mi);

	auto extent = this->getClientExtent();
	int workW = mi.rcWork.right - mi.rcWork.left;
	int workH = mi.rcWork.bottom - mi.rcWork.top;
	int winW = extent.cx;
	int winH = extent.cy;

	int x = mi.rcWork.left + (workW - winW) / 2;
	int y = mi.rcWork.top + (workH - winH) / 2;

	this->setPos(x, y);
}


//!	@brief	Enables or disables the DWM "blur-behind" effect for the window (aka. alpha-composition).
void easywin32::Window::enableBlurBeindWindow(bool enable)
{
	BOOL composition = FALSE;

	if (SUCCEEDED(::DwmIsCompositionEnabled(&composition)) && composition)
	{
		HRGN region = ::CreateRectRgn(0, 0, -1, -1);

		DWM_BLURBEHIND		bb = {};
		bb.dwFlags			= DWM_BB_ENABLE | DWM_BB_BLURREGION;
		bb.hRgnBlur			= enable ? region : NULL;
		bb.fEnable			= enable ? TRUE : FALSE;

		::DwmEnableBlurBehindWindow(m_hWnd, &bb);
		::DeleteObject(region);

		m_enableBlurBeind = enable;
	}
}


/**
 *	@brief		Draws a raw RGB bitmap onto the window at the specified position.
 *	@details	This function copies pixel data directly to the window's client area
 *				using GDI without relying on any GPU or graphics API.
 *	@param[in]	pixels - Pointer to the RGB pixel data in row-major order.
 *	@param[in]	width - Width of the bitmap in pixels.
 *	@param[in]	height - Height of the bitmap in pixels.
 *	@param[in]	dstX - Destination X coordinate in the client area.
 *	@param[in]	dstY - Destination Y coordinate in the client area.
 *	@note		The pixel format must match the expected 24-bit RGB format defined by `Color`.
 *				The function performs an immediate blit to the window using `SetDIBitsToDevice`.
 *	@warning	This function must be called only inside `onPaint`, because it uses BeginPaint/EndPaint.
 *				Calling it outside `onPaint` causes undefined behavior in Win32.
 */
void easywin32::Window::drawBitmap(const ColorRGB * pixels, int width, int height, int dstX, int dstY)
{
	PAINTSTRUCT ps = {};
	BITMAPINFO bmi = {};
	bmi.bmiHeader.biSize			= sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth			= width;
	bmi.bmiHeader.biHeight			= -height;				// Negative = top-down bitmap
	bmi.bmiHeader.biPlanes			= 1;
	bmi.bmiHeader.biBitCount		= sizeof(ColorRGB) * 8;	// 24-bit (RGB)
	bmi.bmiHeader.biCompression		= BI_RGB;				// No compression

	// Begin painting
	HDC hdc = BeginPaint(m_hWnd, &ps);

	// Draw image directly to window
	SetDIBitsToDevice(
		hdc,
		dstX, dstY,			// Destination X, Y
		width, height,		// Width, Height
		0, 0,				// Source X, Y
		0, height,			// Start scan line, number of scan lines
		pixels,				// Pointer to pixel data
		&bmi,
		DIB_RGB_COLORS		// RGB mode
	);

	// End painting
	EndPaint(m_hWnd, &ps);
}


/**
 *	@brief		Waits for and processes the next window message.
 *	@details	This call will block until a new message is available in the queue for this window.
 *				It retrieves the message with `GetMessage`, translates it (e.g. converts virtual-key
 *				messages into character messages), and dispatches it to the window procedure.
 */
void easywin32::Window::waitEvent()
{
	if (m_hWnd != nullptr)
	{
		MSG message = {};

		if (::GetMessage(&message, m_hWnd, 0, 0))
		{
			::TranslateMessage(&message);

			::DispatchMessage(&message);
		};
	}
}


/**
 *	@brief		Processes all pending window messages, if available.
 *	@details	Uses `PeekMessage` to check for all messages in the queue for this window.
 *				If a message exists, it is translated and dispatched to the window procedure.
 *	@return		`true` if a message was processed, `false` if no messages were pending.
 */
bool easywin32::Window::processEvents()
{
	bool hasEvent = false;

	if (m_hWnd != nullptr)
	{
		MSG message = {};

		while (::PeekMessage(&message, m_hWnd, 0, 0, PM_REMOVE))
		{
			::TranslateMessage(&message);

			::DispatchMessage(&message);

			hasEvent = true;
		};
	}

	return hasEvent;
}


/**
 *	@brief		Waits for and processes the next message for all windows belonging to the current thread.
 *	@details	This function blocks until a message is available in the message queue of the
 *				current thread (regardless of which window it belongs to).
 *	@note		Unlike per-window message processing, passing `nullptr` as `hWnd` allows
 *				messages for any window in this thread to be retrieved.
 */
void easywin32::ThreadWindows::waitEvent()
{
	MSG message = {};

	if (::GetMessage(&message, nullptr, 0, 0))
	{
		::TranslateMessage(&message);

		::DispatchMessage(&message);
	};
}


/**
 * @brief      Processes all pending messages, if available.
 * @details    Uses `PeekMessage` to check for all messages in the queue for this thread.
 *             If a message exists, it is translated and dispatched to the window procedure.
 * @return     `true` if an event was processed, `false` if no messages were pending.
 */
bool easywin32::ThreadWindows::processEvents()
{
	MSG message = {};

	bool hasEvent = false;

	while (::PeekMessage(&message, nullptr, 0, 0, PM_REMOVE))
	{
		::TranslateMessage(&message);

		::DispatchMessage(&message);

		hasEvent = true;
	};

	return hasEvent;
}

#endif