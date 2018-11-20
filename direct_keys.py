# direct inputs
# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

escape = 0x01
# 1 = 0x02
# 2 = 0x03
# 3 = 0x04
# 4 = 0x05
# 5 = 0x06
# 6 = 0x07
# 7 = 0x08
# 8 = 0x09
# 9 = 0x0A
# 0 = 0x0B
minus = 0x0C
equals = 0x0D
backspace = 0x0E
tab = 0x0F
Q = 0x10
W = 0x11
E = 0x12
R = 0x13
T = 0x14
Y = 0x15
U = 0x16
I = 0x17
O = 0x18
P = 0x19
leftbracket = 0x1A
rightbracket = 0x1B
enter = 0x1C
leftcontrol = 0x1D
A = 0x1E
S = 0x1F
D = 0x20
F = 0x21
G = 0x22
H = 0x23
J = 0x24
K = 0x25
L = 0x26
semicolon = 0x27
apostrophe = 0x28
# ~(console) = 0x29
leftshift = 0x2A
backslash = 0x2B
Z = 0x2C
X = 0x2D
C = 0x2E
V = 0x2F
B = 0x30
N = 0x31
M = 0x32
comma = 0x33
period = 0x34
forwardslash = 0x35
rightshift = 0x36
# num* = 0x37
leftalt = 0x38
spacebar = 0x39
capslock = 0x3A
f1 = 0x3B
f2 = 0x3C
f3 = 0x3D
f4 = 0x3E
f5 = 0x3F
f6 = 0x40
f7 = 0x41
f8 = 0x42
f9 = 0x43
f10 = 0x44
numlock = 0x45
scrolllock = 0x46
num7 = 0x47
num8 = 0x48
num9 = 0x49
# num- = 0x4A
num4 = 0x4B
num5 = 0x4C
num6 = 0x4D
# num+ = 0x4E
num1 = 0x4F
num2 = 0x50
num3 = 0x51
num0 = 0x52
# num. = 0x53
f11 = 0x57
f12 = 0x58
numenter = 0x9C
rightcontrol = 0x9D
# num/ = 0xB5
rightalt = 0xB8
home = 0xC7
uparrow = 0xC8
pgup = 0xC9
leftarrow = 0xCB
rightarrow = 0xCD
end = 0xCF
downarrow = 0xD0
pgdown = 0xD1
insert = 0xD2
delete = 0xD3
leftmousebutton = 0x100
rightmousebutton = 0x101
# middle/wheelmousebutton = 0x102
mousebutton3 = 0x103
mousebutton4 = 0x104
mousebutton5 = 0x105
mousebutton6 = 0x106
mousebutton7 = 0x107
mousewheelup = 0x108
mousewheeldown = 0x109

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


if __name__ == '__main__':
    PressKey(0x11)
    time.sleep(1)
    ReleaseKey(0x11)
    time.sleep(1)
