#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
import time, os, random, argparse, sys, math
from enum import Enum, auto
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import signal

import re

def dbg(*a):
    if DEBUG:
        # Convert enum values to strings for better readability in debug output
        formatted_args = []
        for arg in a:
            if isinstance(arg, NekoState):
                # Show both enum name and string representation for better debugging
                formatted_args.append(f"{arg.name}")
            else:
                formatted_args.append(arg)
        print(time.strftime("[%H:%M:%S]"), *formatted_args, flush=True)

def err(*a):
    formatted_args = []
    for arg in a:
        formatted_args.append(arg)
    print(time.strftime("[%H:%M:%S]"), *formatted_args, flush=True)
    exit(1)


@dataclass
class ScreenData:
    WindowWidth:int
    WindowHeight:int
    background_image: Image.Image | None = None
    MouseX:int = 0
    MouseY:int = 0
    PrevMouseX:int = 0
    PrevMouseY:int = 0


# States enum for neko (adapted from oneko.c NEKO_STATE)
class NekoState(Enum):
    # Rest states (stationary positions)
    STOP = auto()      # NEKO_STOP
    JARE = auto()      # NEKO_JARE (grooming)
    KAKI = auto()      # NEKO_KAKI (scratching)
    AKUBI = auto()     # NEKO_AKUBI (yawning)
    SLEEP = auto()     # NEKO_SLEEP
    AWAKE = auto()     # NEKO_AWAKE
    
    # Movement states
    UP = auto()        # NEKO_U_MOVE
    DOWN = auto()      # NEKO_D_MOVE
    LEFT = auto()      # NEKO_L_MOVE
    RIGHT = auto()     # NEKO_R_MOVE
    UPLEFT = auto()    # NEKO_UL_MOVE
    UPRIGHT = auto()   # NEKO_UR_MOVE
    DWLEFT = auto()    # NEKO_DL_MOVE
    DWRIGHT = auto()   # NEKO_DR_MOVE
    
    # Togi states (scratching states)
    UTOGI = auto()     # NEKO_U_TOGI (up scratching)
    DTOGI = auto()     # NEKO_D_TOGI (down scratching)
    LTOGI = auto()     # NEKO_L_TOGI (left scratching)
    RTOGI = auto()     # NEKO_R_TOGI (right scratching)

# Animation durations as constants (from oneko.h)
# These mirror the original names for clarity
NEKO_STOP_TIME  = 4
NEKO_JARE_TIME  = 10
NEKO_KAKI_TIME  = 4
NEKO_AKUBI_TIME = 6
NEKO_AWAKE_TIME = 3
NEKO_TOGI_TIME  = 10

@dataclass
class CursorImage:
    name: str
    sprite: Image.Image | None = None
    mask: Image.Image | None = None
    x_hot: int | None = None
    y_hot: int | None = None


# @dataclass
# class CursorData
#     MouseX:int = 0
#     MouseY:int = 0
#     PrevMouseX:int = 0
#     PrevMouseY:int = 0
#     cursor: CursorImage


@dataclass
class Animation:
    sprite: Image.Image
    mask: Image.Image

@dataclass
class AnimalData:
    NekoSpeed: int #Movement speed
    IdleSpace: int #Defines the "dead zone" around cursor - neko won't move if cursor stays within this area
    IntervalTime: int
    XOffset: int    # X and Y offsets for cat from mouse pointer.
    YOffset: int
    cursor: CursorImage
    color = None
    edge_color = None
    bitmap_width = 0
    bitmap_height = 0

    AnimationPattern: dict[NekoState, list[Animation]] = None
    Duartions = {
    NekoState.SLEEP: 0,            # Sleep duration (not defined in oneko.h)
    NekoState.STOP: float(NEKO_STOP_TIME),
    NekoState.JARE: float(NEKO_JARE_TIME),
    NekoState.KAKI: float(NEKO_KAKI_TIME),
    NekoState.AKUBI: float(NEKO_AKUBI_TIME),
    NekoState.AWAKE: float(NEKO_AWAKE_TIME),
    NekoState.LTOGI: float(NEKO_TOGI_TIME),
    NekoState.RTOGI: float(NEKO_TOGI_TIME),
    NekoState.UTOGI: float(NEKO_TOGI_TIME),
    NekoState.DTOGI: float(NEKO_TOGI_TIME),
    }
    state = NekoState.STOP

    NekoX, NekoY = 0, 0
    NekoLastX, NekoLastY = 0, 0
    NekoMoveDx, NekoMoveDy = 0, 0

    # Global tick counter (frames, animation steps), used for switching animation frames
    NekoTickCount = 0
    # Time counter in current state, incremented only on every second tick
    NekoStateCount = 0


cursor_names = ["mouse", "bone", "card", "petal"]

# Animal parameters table
# Map pet name to AnimalData for fast lookup
AnimalDefaultsDataTable = {
    "neko":   AnimalData(13, 6, 125000, 0, 0, CursorImage("mouse")),
    "tora":   AnimalData(16, 6, 125000, 0, 0, CursorImage("mouse")),
    "dog":    AnimalData(10, 6, 125000, 0, 0, CursorImage("bone")),
    "sakura": AnimalData(13, 6, 125000, 0, 0, CursorImage("card")),
    "tomoyo": AnimalData(10, 6, 125000, 32, 32, CursorImage("petal")),
}

# Animation patterns based on oneko.c AnimationPattern structure
ANIMATION_PATTERNS = {
    NekoState.STOP:      ["mati2", "mati2"],        # NEKO_STOP
    NekoState.JARE:      ["jare2", "mati2"],        # NEKO_JARE  
    NekoState.KAKI:      ["kaki1", "kaki2"],       # NEKO_KAKI
    NekoState.AKUBI:     ["mati3", "mati3"],       # NEKO_AKUBI
    NekoState.SLEEP:     ["sleep1", "sleep2"],     # NEKO_SLEEP
    NekoState.AWAKE:     ["awake", "awake"],       # NEKO_AWAKE
    NekoState.UP:        ["up1", "up2"],           # NEKO_U_MOVE
    NekoState.DOWN:      ["down1", "down2"],       # NEKO_D_MOVE
    NekoState.LEFT:      ["left1", "left2"],       # NEKO_L_MOVE
    NekoState.RIGHT:     ["right1", "right2"],     # NEKO_R_MOVE
    NekoState.UPLEFT:    ["upleft1", "upleft2"],   # NEKO_UL_MOVE
    NekoState.UPRIGHT:   ["upright1", "upright2"], # NEKO_UR_MOVE
    NekoState.DWLEFT:    ["dwleft1", "dwleft2"],   # NEKO_DL_MOVE
    NekoState.DWRIGHT:   ["dwright1", "dwright2"], # NEKO_DR_MOVE
    NekoState.UTOGI:     ["utogi1", "utogi2"],     # NEKO_U_TOGI
    NekoState.DTOGI:     ["dtogi1", "dtogi2"],     # NEKO_D_TOGI
    NekoState.LTOGI:     ["ltogi1", "ltogi2"],     # NEKO_L_TOGI
    NekoState.RTOGI:     ["rtogi1", "rtogi2"],     # NEKO_R_TOGI
}


# MOVE_DIRS = {
#     NekoState.LEFT: (-1,0), NekoState.RIGHT: (1,0), NekoState.UP: (0,-1), NekoState.DOWN: (0,1),
#     NekoState.UPLEFT: (-1,-1), NekoState.UPRIGHT: (1,-1), NekoState.DWLEFT: (-1,1), NekoState.DWRIGHT: (1,1)
# }


# States that don't involve movement (rest states)
REST_STATES = {NekoState.SLEEP, NekoState.STOP, NekoState.JARE, NekoState.KAKI, NekoState.AKUBI, NekoState.AWAKE, 
               NekoState.LTOGI, NekoState.RTOGI, NekoState.UTOGI, NekoState.DTOGI}

# Movement states
MOVE_STATES = {
    NekoState.UP, NekoState.DOWN, NekoState.LEFT, NekoState.RIGHT,
    NekoState.UPLEFT, NekoState.UPRIGHT, NekoState.DWLEFT, NekoState.DWRIGHT
}

# Scratch (togi) states
SCRATCH_STATES = {
    NekoState.UTOGI, NekoState.DTOGI, NekoState.LTOGI, NekoState.RTOGI
}

DEBUG = False


# Constants from oneko.h
MAX_TICK = 9999  # Odd Only! (maximum value for counters)

DEFAULT_BACKGROUND_COLOR = (50, 50, 50)

DEFAULT_MATRIX_ROWS = 64
DEFAULT_MATRIX_COLS = 64

SinPiPer8 = math.sin(math.pi/8.0)
SinPiPer8Times3 = math.sin(3*math.pi/8.0)

# Global sprite dimensions - will be initialized when loading first .xbm file
DEFAULT_SPRITE_COLOR = (255,255,255)
DEFAULT_SPRITE_EDGE_COLOR = (0,0,0)

DEFAULT_ANIMAL = "neko"

pet: AnimalData | None = None
screen :ScreenData | None = None

background_image_path = "backgrounds/grass4.png"
sprites_path = ""
masks_path = ""
sprite_file_suffix = ""
mask_file_suffix = ""

def GetArguments():
    global DEFAULT_MATRIX_ROWS, DEFAULT_MATRIX_COLS, DEFAULT_SPRITE_COLOR, DEFAULT_SPRITE_EDGE_COLOR, DEFAULT_BACKGROUND_COLOR, background_image_path

    parser = argparse.ArgumentParser(description="Neko on rpi-rgb-led-rgb_matrix with XBM and masks")
    animal_names = list(AnimalDefaultsDataTable.keys())
    parser.add_argument("--name", default=DEFAULT_ANIMAL, choices=animal_names, help="Animal name (neko, tora, dog, sakura, tomoyo)")
    parser.add_argument("--time",    type=int,
                        help="Timer interval in microseconds (default: from pet table, like oneko -time)")
    parser.add_argument("--speed",   type=int,
                        help="Neko running speed (default: from pet table, like oneko -speed)")
    parser.add_argument("--idle-space", type=int,
                       help="Idle space around cursor (default: from pet table, like in oneko.c)")
               
    parser.add_argument("--sprite-color", type=str, default=",".join(map(str, DEFAULT_SPRITE_COLOR)),
                       help="Sprite color as R,G,B (default: %(default)s)")
    parser.add_argument("--sprite-edge-color", type=str, default=",".join(map(str, DEFAULT_SPRITE_EDGE_COLOR)),
                       help="Sprite color as R,G,B (default: %(default)s)")    

    parser.add_argument("--background-image", type=str, default=background_image_path,
                       help="Path to background sprite (PNG/JPG). (default: %(default)s)")
    parser.add_argument("--background-color", type=str, default=",".join(map(str, DEFAULT_BACKGROUND_COLOR)),
                        help="Background color as R,G,B (default: %(default)s)")
    parser.add_argument("--position", type=str, help="Initial position as X,Y (like oneko -position)")

    parser.add_argument("--rows", type=int, default=DEFAULT_MATRIX_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_MATRIX_COLS)
    parser.add_argument("--debug",   action="store_true")

    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug


    global screen
    # Save rgb_matrix size to globals (only from arguments or defaults, not from pet)
    screen=ScreenData(args.cols, args.rows)

    global pet
    # Find animal parameters by name using map
    pet = AnimalDefaultsDataTable.get(args.name)
    if pet is None:
        err(f"Error: No such pet {args.name}!")

    # Update global IdleSpace from command line arguments or from pet
    pet.IdleSpace = args.idle_space if args.idle_space is not None else pet.IdleSpace

    # Apply speed to global NekoSpeed (movement per tick)
    pet.NekoSpeed = float(args.speed if args.speed is not None else pet.NekoSpeed)

    # Emulate oneko's -time (microseconds) using sleep in seconds
    # If args.time is not set, use value from pet.time
    pet.IntervalTime = args.time if args.time is not None else pet.IntervalTime

    # parse colors
    pet.color = tuple(map(int, args.sprite_color.split(",")))
    pet.edge_color = tuple(map(int, args.sprite_edge_color.split(",")))
    DEFAULT_BACKGROUND_COLOR = tuple(map(int, args.background_color.split(",")))

    background_image_path = args.background_image

    global sprites_path, masks_path, sprite_file_suffix, mask_file_suffix
    # Define folder and suffix for sprites
    sprites_path = f"bitmaps/{args.name}"
    if args.name == "neko":
        sprite_file_suffix = ""
    else:
        sprite_file_suffix = f"_{args.name}"

    # Define folder and suffix for masks
    if args.name == "tora":
        masks_path = "bitmasks/neko"
        mask_file_suffix = ""
    elif args.name == "neko":
        masks_path = "bitmasks/neko"
        mask_file_suffix = ""
    else:
        masks_path = f"bitmasks/{args.name}"
        mask_file_suffix = f"_{args.name}"

    if args.position:
        try:
            x, y = map(int, args.position.split(","))
            pet.XOffset = x + pet.XOffset;
            pet.YOffset = y + pet.YOffset;
        except Exception as e:
            err(f"Error parsing --position: {e}")


# ---- Load cursor images from cursors/ folder ----
def LoadCursor():
    def ParseHotSpot(xbm_path: str):
        x_hot, y_hot = None, None
        try:
            with open(xbm_path, "r") as f:
                text = f.read()
            m = re.search(r"#define\s+\w+_x_hot\s+(\d+)", text)
            if m:
                x_hot = int(m.group(1))
            m = re.search(r"#define\s+\w+_y_hot\s+(\d+)", text)
            if m:
                y_hot = int(m.group(1))
        except Exception as e:
            err(f"Error reading hotspot from {xbm_path}: {e}")
        return x_hot, y_hot

    global pet
    bits = f"{pet.cursor.name}_cursor.xbm"
    mask = f"{pet.cursor.name}_cursor_mask.xbm"
    bits_path = os.path.join("cursors", bits)
    mask_path = os.path.join("cursors", mask)
    try:
        bits_img = Image.open(bits_path).convert("1")
        mask_img = Image.open(mask_path).convert("1")
        x_hot, y_hot = ParseHotSpot(bits_path)
        pet.cursor = CursorImage(pet.cursor.name, bits_img, mask_img, x_hot, y_hot)
    except Exception as e:
        err(f"Error loading cursor {pet.cursor.name}: {e}")

# ---- Load background ----
def LoadBackground():
    global DEFAULT_BACKGROUND_COLOR, background_image_path, screen
    try:
        screen.background_image = Image.open(background_image_path).convert('RGB').resize((screen.WindowWidth, screen.WindowHeight), Image.Resampling.LANCZOS)
    except Exception as e:
        dbg(f"Error loading background: {e}")
        screen.background_image = Image.new('RGB', (screen.WindowWidth, screen.WindowHeight), DEFAULT_BACKGROUND_COLOR)

# ---- Load animations ----
def LoadAnimations():
    global sprites_path, masks_path, pet
    animations = {}
    first_sprite_size = None
    for anim_name, basenames in ANIMATION_PATTERNS.items():
        loaded = []
        for base in basenames:
            sprite_base = f"{base}{sprite_file_suffix}"
            sp = os.path.join(sprites_path, f"{sprite_base}.xbm")
            mask_base = f"{base}{mask_file_suffix}_mask"
            mp = os.path.join(masks_path, f"{mask_base}.xbm")
            if not (os.path.exists(sp) and os.path.exists(mp)):
                continue
            sprite = Image.open(sp).convert("1")
            mask   = Image.open(mp).convert("1")
            # Set bitmap_width and bitmap_height from the first loaded sprite
            if first_sprite_size is None:
                first_sprite_size = sprite.size
                pet.bitmap_width = sprite.size[0]
                pet.bitmap_height = sprite.size[1]
            # Validate sprite dimensions match pet's expected dimensions
            if sprite.size[0] != pet.bitmap_width or sprite.size[1] != pet.bitmap_height:
                dbg(f"Warning: sprite {sprite_base}.xbm has dimensions {sprite.size[0]}x{sprite.size[1]}, expected {pet.bitmap_width}x{pet.bitmap_height}")
                continue
            if mask.size[0] != pet.bitmap_width or mask.size[1] != pet.bitmap_height:
                dbg(f"Warning: mask {mask_base}.xbm has dimensions {mask.size[0]}x{mask.size[1]}, expected {pet.bitmap_width}x{pet.bitmap_height}")
                continue
            loaded.append(Animation(sprite, mask))
        if loaded:
            animations[anim_name] = loaded
    if animations:
        dbg(f"Loaded {len(animations)} animations with sprite dimensions {pet.bitmap_width}x{pet.bitmap_height}")
        pet.AnimationPattern = animations
    else:
        err("No sprites were loaded!")

# ===========================================================================





# --- Base Display Interface ---
class BaseDisplay(ABC):
    running = True
    def __init__(self, screen: ScreenData):
        if not screen:
            raise ValueError("Screen configuration is required")
        self.screen = screen
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print("\nCatched SIGINT, exit...")
        self.running = False

    @abstractmethod
    def ProcessEvent(self) -> bool: ...
    @abstractmethod
    def FreeScreen(self) -> None: ...
    @abstractmethod
    def InitMatrix(self) -> None: ...
    @abstractmethod
    def UpdateScreen(self) -> None: ...
    @abstractmethod
    def SetPixel(self, x: int, y: int, color: tuple[int, int, int]) -> None: ...
    @abstractmethod
    def SetImage(self, image: Image.Image, x: int, y: int) -> None: ...


# --- Global display instance ---
global_display: BaseDisplay | None = None

def CreateDisplay(screen: ScreenData, scale: int = 8) -> BaseDisplay:
    try:
        return RGBMatrixDisplay(screen)
    except ImportError:
        try:
            return PygameDisplay(screen, scale=scale)
        except ImportError:
            err("❌ No rpi-rgb-led-rgb_matrix or no pygame")


# --- RGBMatrix Implementation ---
class RGBMatrixDisplay(BaseDisplay):
    def __init__(self, screen: ScreenData):
        super().__init__(screen)
        from rgbmatrix import RGBMatrix, RGBMatrixOptions
        self.RGBMatrix = RGBMatrix
        self.RGBMatrixOptions = RGBMatrixOptions
        self.rgb_matrix = None
        self.double_buffer = None

    def ProcessEvent(self) -> bool:
        return self.running

    def FreeScreen(self) -> None:
        self.rgb_matrix.Clear()
        self.rgb_matrix = None

    def InitMatrix(self) -> None:
        opts = self.RGBMatrixOptions()
        opts.rows = self.screen.WindowHeight
        opts.cols = self.screen.WindowWidth
        opts.chain_length = 1
        opts.parallel = 1
        opts.hardware_mapping = "regular"
        opts.gpio_slowdown = 2
        opts.row_address_type = 0
        opts.pwm_bits = 11
        opts.brightness = 100
        opts.pwm_lsb_nanoseconds = 80
        opts.show_refresh_rate = 1
        opts.limit_refresh_rate_hz = 100
        opts.scan_mode = 0
        opts.disable_busy_waiting = 1

        self.rgb_matrix = self.RGBMatrix(options=opts)
        self.double_buffer = self.rgb_matrix.CreateFrameCanvas()

    def UpdateScreen(self) -> None:
        self.double_buffer = self.rgb_matrix.SwapOnVSync(self.double_buffer)

    def SetPixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        self.double_buffer.SetPixel(x, y, *color)

    def SetImage(self, image: Image.Image, x: int, y: int) -> None:
        self.double_buffer.SetImage(image, x, y)


# --- Pygame Implementation ---
class PygameDisplay(BaseDisplay):

    def __init__(self, screen: ScreenData, scale: int = 8, show_grid: bool = True):
        super().__init__(screen)
        import pygame
        self.pygame = pygame
        self.display = None
        self.double_buffer = None
        self.SCALE = scale
        self.show_grid = show_grid

    def ProcessEvent(self) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            elif event.type == self.pygame.MOUSEBUTTONDOWN:
                UpdatePointerPos(event.pos[0] // self.SCALE, event.pos[1] // self.SCALE)
        return self.running

    def FreeScreen(self) -> None:
        self.pygame.quit()

    def InitMatrix(self) -> None:
        self.pygame.init()
        self.display = self.pygame.display.set_mode(
            (self.screen.WindowWidth * self.SCALE, self.screen.WindowHeight * self.SCALE)
        )
        self.pygame.display.set_caption(f"Oneko led rgb_matrix emu {self.screen.WindowWidth}x{self.screen.WindowHeight}")
        self.double_buffer = self.pygame.Surface((self.screen.WindowWidth, self.screen.WindowHeight))

    def UpdateScreen(self) -> None:
        scaled = self.pygame.transform.scale(
            self.double_buffer,
            (self.screen.WindowWidth * self.SCALE, self.screen.WindowHeight * self.SCALE)
        )
        self.display.blit(scaled, (0, 0))
        if self.show_grid and self.SCALE > 2:
            self._draw_grid()
        self.pygame.display.flip()

    def _draw_grid(self) -> None:
        grid_color = (0, 0, 0)
        for x in range(self.screen.WindowWidth + 1):
            xpos = x * self.SCALE
            self.pygame.draw.line(self.display, grid_color, (xpos, 0), (xpos, self.screen.WindowHeight * self.SCALE))
        for y in range(self.screen.WindowHeight + 1):
            ypos = y * self.SCALE
            self.pygame.draw.line(self.display, grid_color, (0, ypos), ( self.screen.WindowWidth * self.SCALE, ypos))

    def SetPixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        self.double_buffer.set_at((x, y), color)

    def SetImage(self, image: Image.Image, x: int, y: int) -> None:
        img = image.convert('RGBA')
        try:
            py_image = self.pygame.image.fromstring(img.tobytes(), img.size, 'RGBA')
            self.double_buffer.blit(py_image, (x, y))
        except Exception as e:
            print(f"Error converting image: {e}")

# ===========================================================================

def Drawsprite(frame, sprite, mask, x0, y0, color, edge_color):
    rgba_sprite = Image.composite(
        Image.new("RGBA", (sprite.width, sprite.height), color),
        Image.new("RGBA", (sprite.width, sprite.height), edge_color),
        sprite.convert("L").point(lambda v: 255 if v == 0 else 0)
    )
    rgba_sprite.putalpha(mask)
    frame.paste(rgba_sprite, (x0, y0), rgba_sprite)

# def move_virtual_cursor_smoothly(target_x, target_y, steps=20, delay=0.01):
#     """Move the virtual cursor smoothly from current position to target position."""
#     global screen.MouseX, screen.MouseY
#     start_x, start_y = screen.MouseX, screen.MouseY
#     for i in range(1, steps+1):
#         new_x = int(start_x + (target_x - start_x) * i / steps)
#         new_y = int(start_y + (target_y - start_y) * i / steps)
#         screen.MouseX = new_x
#         screen.MouseY = new_y
#         dbg(f"move_virtual_cursor_smoothly: step {i}/{steps} -> ({screen.MouseX}, {screen.MouseY})")
#         update_screen()
#         time.sleep(delay)

VirtualCursorTimer = 0
VirtualCursorInterval = random.randint(50, 100)  # Random interval between 5-10 seconds


PointerPosX=0
PointerPosY=0

def UpdatePointerPos(x, y):
    global PointerPosX, PointerPosY, VirtualCursorTimer, screen
    cursor_w, cursor_h = pet.cursor.sprite.size
    x_hot, y_hot = pet.cursor.x_hot, pet.cursor.y_hot
    # Ensure the hotspot stays within the rgb_matrix
    min_x = x_hot
    min_y = y_hot
    max_x = screen.WindowWidth - (cursor_w - x_hot)
    max_y = screen.WindowHeight - (cursor_h - y_hot)
    PointerPosX = max(min_x, min(max_x, x))
    PointerPosY = max(min_y, min(max_y, y))
    VirtualCursorTimer = 0
    dbg(f"Mouse clicked at: ({x}, {y}) -> VirtualMouse: ({PointerPosX}, {PointerPosY})")

def updatePointerPosRandomly():
    """Move the virtual cursor to a random position within bounds (cursor fully visible)"""
    global pet, screen 
    global PointerPosX, PointerPosY
    cursor_w, cursor_h = pet.cursor.sprite.size
    x_hot, y_hot = pet.cursor.x_hot, pet.cursor.y_hot
    # Ensure the hotspot stays within the rgb_matrix
    min_x = x_hot
    min_y = y_hot
    max_x = screen.WindowWidth - (cursor_w - x_hot)
    max_y = screen.WindowHeight - (cursor_h - y_hot)
    PointerPosX = random.randint(min_x, max_x)
    PointerPosY = random.randint(min_y, max_y)
    print(f"move_virtual_cursor_randomly: moved to ({PointerPosX}, {PointerPosY}), hotspot=({x_hot},{y_hot})")

def QueryPointer():
    """Virtual cursor simulation - changes position every 5-10 seconds"""
    global VirtualCursorTimer, VirtualCursorInterval

    VirtualCursorTimer += 1
    if VirtualCursorTimer >= VirtualCursorInterval:
        updatePointerPosRandomly()
        # Reset timer and generate new random interval
        VirtualCursorTimer = 0
        VirtualCursorInterval = random.randint(50, 100)
        dbg(f"Virtual cursor moved to ({PointerPosX}, {PointerPosY})")

    return PointerPosX, PointerPosY

def GetResources():
    LoadAnimations()
    LoadBackground()
    LoadCursor()

def InitScreen():
    GetResources()

    global global_display
    global_display=CreateDisplay(screen)
    global_display.InitMatrix()

def FreeScreen():
    global global_display
    global_display.FreeScreen()

def ProcessEvent():
    global global_display
    return global_display.ProcessEvent()

def LoadPlugins():
    global pet, screen, plugins
    try:
        from plugins.plugin_base import load_plugins
        plugins = load_plugins(pet, screen)
        for p in plugins:
            print(f"Starting plugin {p.name()}")
            p.start()
    except ImportError:
        plugins = None
        print("No plugins")        

def StopPlugins():  
    global plugins
    if plugins:
        for p in plugins:
            p.stop()      


def DrawNeko(x, y, animation):
    global  pet, screen, global_display
    frame = screen.background_image.copy().convert("RGB")

    Drawsprite(frame, animation.sprite, animation.mask, x, y, pet.color, pet.edge_color)
    if pet.state==NekoState.AWAKE or pet.state not in REST_STATES:
        cursor_x = screen.MouseX - pet.cursor.x_hot
        cursor_y = screen.MouseY - pet.cursor.y_hot
        Drawsprite(frame, pet.cursor.sprite, pet.cursor.mask, cursor_x, cursor_y, pet.color, pet.edge_color)

    global_display.SetImage(frame, 0, 0)
    global_display.UpdateScreen()

    pet.NekoLastX = x
    pet.NekoLastY = y

# ===========================================================================

# original oneko.c logic
def SetNekoState(new_state):
    global pet
    old_state = pet.state

    # Reset counters when changing state
    pet.NekoTickCount = 0
    pet.NekoStateCount = 0

    pet.state = new_state
    dbg(f"STATE: {old_state} -> {new_state}")

# original oneko.c logic
def NekoDirection():
    global pet
    if pet.NekoMoveDx == 0 and pet.NekoMoveDy == 0:
        NewState = NekoState.STOP
    else:
        LargeX = float(pet.NekoMoveDx)
        LargeY = float(-pet.NekoMoveDy)
        Length = math.sqrt(LargeX*LargeX + LargeY*LargeY)
        # if Length == 0:
        #     NewState = NekoState.STOP
        # else:
        SinTheta = LargeY/Length
        if pet.NekoMoveDx > 0:
            if SinTheta > SinPiPer8Times3:
                NewState = NekoState.UP
            elif SinTheta > SinPiPer8:
                NewState = NekoState.UPRIGHT
            elif SinTheta > -(SinPiPer8):
                NewState = NekoState.RIGHT
            elif SinTheta > -(SinPiPer8Times3):
                NewState = NekoState.DWRIGHT
            else:
                NewState = NekoState.DOWN
        else:
            if SinTheta > SinPiPer8Times3:
                NewState = NekoState.UP
            elif SinTheta > SinPiPer8:
                NewState = NekoState.UPLEFT
            elif SinTheta > -(SinPiPer8):
                NewState = NekoState.LEFT
            elif SinTheta > -(SinPiPer8Times3):
                NewState = NekoState.DWLEFT
            else:
                NewState = NekoState.DOWN
    
    # Only change state if we have a valid new state and it's different
    if pet.state != NewState:
        dbg(f"NekoDirection: {pet.state} -> {NewState} (dx={pet.NekoMoveDx}, dy={pet.NekoMoveDy})")
        SetNekoState(NewState)


# original oneko.c logic
def CalcDxDy(): #Calculates movement direction based on cursor position
    global pet, screen

    # Save previous cursor position
    screen.PrevMouseX = screen.MouseX
    screen.PrevMouseY = screen.MouseY

    # Update virtual cursor
    AbsoluteX, AbsoluteY = QueryPointer()

    screen.MouseX = AbsoluteX + pet.XOffset
    screen.MouseY = AbsoluteY + pet.YOffset

    LargeX = float(screen.MouseX - (pet.NekoX + pet.bitmap_width / 2))
    LargeY = float(screen.MouseY - (pet.NekoY + pet.bitmap_height / 2))
    # In the original code, Y is calculated using full sprite height
    #LargeX = float(screen.MouseX - pet.NekoX - pet.bitmap_width / 2)
    #LargeY = float(screen.MouseY - pet.NekoY - pet.bitmap_height)

    DoubleLength = LargeX * LargeX + LargeY * LargeY

    if DoubleLength != 0.0:
        Length = math.sqrt(DoubleLength)

        if Length <= pet.NekoSpeed:
            pet.NekoMoveDx = int(LargeX)
            pet.NekoMoveDy = int(LargeY)
        else:
            pet.NekoMoveDx = int(round((pet.NekoSpeed * LargeX) / Length))
            pet.NekoMoveDy = int(round((pet.NekoSpeed * LargeY) / Length))

        dbg(f"CalcDxDy: neko({pet.NekoX},{pet.NekoY}) -> cursor({screen.MouseX},{screen.MouseY}) = dx:{pet.NekoMoveDx}, dy:{pet.NekoMoveDy}, Length:{Length:.2f}")
    else:
        pet.NekoMoveDx = pet.NekoMoveDy = 0
        dbg("CalcDxDy: neko at cursor position, no movement needed")

# original oneko.c logic
def IsNekoDontMove(): #Checks if neko has moved from last position (used in boundary detection)
    """Check if neko has moved from last position (adapted from oneko.c)"""
    if pet.NekoX == pet.NekoLastX and pet.NekoY == pet.NekoLastY:
        dbg(f"IsNekoDontMove: neko hasn't moved (pos: {pet.NekoX},{pet.NekoY}, last: {pet.NekoLastX},{pet.NekoLastY})")
        return True
    else:
        dbg(f"IsNekoDontMove: neko has moved (pos: {pet.NekoX},{pet.NekoY}, last: {pet.NekoLastX},{pet.NekoLastY})")
        return False

# original oneko.c logic
def IsNekoMoveStart():
    # Check if virtual cursor has moved outside the IdleSpace (dead zone)
    # This is the exact logic from oneko.c
    if ((screen.PrevMouseX >= screen.MouseX - pet.IdleSpace
         and screen.PrevMouseX <= screen.MouseX + pet.IdleSpace) and
        (screen.PrevMouseY >= screen.MouseY - pet.IdleSpace
         and screen.PrevMouseY <= screen.MouseY + pet.IdleSpace)):
        dbg(f"IsNekoMoveStart: cursor in idle zone (prev: {screen.PrevMouseX},{screen.PrevMouseY}, current: {screen.MouseX},{screen.MouseY}, IdleSpace: {pet.IdleSpace})")
        return False  # Cursor hasn't moved enough to trigger movement
    else:
        dbg(f"IsNekoMoveStart: cursor moved outside idle zone (prev: {screen.PrevMouseX},{screen.PrevMouseY}, current: {screen.MouseX},{screen.MouseY}, IdleSpace: {pet.IdleSpace})")
        return True   # Cursor has moved significantly, start movement

# original oneko.c logic
def IsWindowOver():
    # Check if Neko is outside screen bounds and clamp position (adapted from oneko.c)
    global pet, screen
    ReturnValue = False

    if pet.NekoY <= 0:
        pet.NekoY = 0
        ReturnValue = True
    elif pet.NekoY >= screen.WindowHeight - pet.bitmap_height:
        pet.NekoY = screen.WindowHeight - pet.bitmap_height
        ReturnValue = True

    if pet.NekoX <= 0:
        pet.NekoX = 0
        ReturnValue = True
    elif pet.NekoX >=  screen.WindowWidth - pet.bitmap_width:
        pet.NekoX = screen.WindowWidth - pet.bitmap_width
        ReturnValue = True

    return ReturnValue


# original oneko.c logic
def Interval():
    time.sleep(pet.IntervalTime / 1_000_000.0)


# original oneko.c logic
def TickCount():
    # Tick counter management adapted from oneko.c TickCount() function
    global pet
    
    # Increment NekoTickCount and wrap around at MAX_TICK
    pet.NekoTickCount += 1
    if pet.NekoTickCount >= MAX_TICK:
        pet.NekoTickCount = 0
    
    # Increment NekoStateCount every 2 ticks (only if below MAX_TICK)
    if pet.NekoTickCount % 2 == 0:
        if pet.NekoStateCount < MAX_TICK:
            pet.NekoStateCount += 1
    
    dbg(f"TickCount: NekoTickCount={pet.NekoTickCount}, NekoStateCount={pet.NekoStateCount}")

# original oneko.c logic
def NekoThinkDraw():
    global pet, screen

    # Then calculate movement direction
    CalcDxDy()

    if pet.state != NekoState.SLEEP:
        DrawNeko(pet.NekoX, pet.NekoY,
                 pet.AnimationPattern[pet.state][pet.NekoTickCount & 0x1]);
    else:
        DrawNeko(pet.NekoX, pet.NekoY,
                 pet.AnimationPattern[pet.state][(pet.NekoTickCount >> 2) & 0x1]);

    TickCount()
    
    if pet.NekoTickCount % 10 == 0:  # Log every 10 frames to avoid spam
        dbg(f"Frame: state={pet.state}, NekoStateCount={pet.NekoStateCount}, NekoTickCount={pet.NekoTickCount}")

    # State machine logic adapted from oneko.c using match/case
    match pet.state:
        case NekoState.STOP:
            if IsNekoMoveStart():
                dbg(f"stop -> awake (cursor movement detected)")
                SetNekoState(NekoState.AWAKE)
            elif pet.NekoStateCount < pet.Duartions[pet.state]:
                dbg(f"stop: waiting, count={pet.NekoStateCount}/{pet.Duartions[pet.state]}")
                pass  # Continue in current state
            else:
                # After stop time expires, move to next state regardless of cursor movement
                if pet.NekoMoveDx < 0 and pet.NekoX <= 0:
                    SetNekoState(NekoState.LTOGI)
                elif pet.NekoMoveDx > 0 and pet.NekoX >= screen.WindowWidth - pet.bitmap_width:
                    SetNekoState(NekoState.RTOGI)
                elif (pet.NekoMoveDy < 0 and pet.NekoY <= 0):
                    SetNekoState(NekoState.UTOGI)
                elif (pet.NekoMoveDy > 0 and pet.NekoY >= screen.WindowWidth - pet.bitmap_height):
                    SetNekoState(NekoState.DTOGI)
                else:
                    SetNekoState(NekoState.JARE)

        case NekoState.JARE:
            if IsNekoMoveStart():
                dbg(f"jare -> awake (cursor movement detected)")
                SetNekoState(NekoState.AWAKE)
            elif pet.NekoStateCount < pet.Duartions[pet.state]:
                dbg(f"jare: waiting, count={pet.NekoStateCount}/{pet.Duartions[pet.state]}")
                pass
            else:
                dbg(f"jare -> kaki (time expired)")
                SetNekoState(NekoState.KAKI)

        case NekoState.KAKI:
            if IsNekoMoveStart():
                dbg(f"kaki -> awake (cursor movement detected)")
                SetNekoState(NekoState.AWAKE)
            elif pet.NekoStateCount < pet.Duartions[pet.state]:
                dbg(f"jare: waiting, count={pet.NekoStateCount}/{pet.Duartions[pet.state]}")
                pass
            else:
                dbg(f"kaki -> akubi (time expired)")
                SetNekoState(NekoState.AKUBI)

        case NekoState.AKUBI:
            if IsNekoMoveStart():
                dbg(f"akubi -> awake (cursor movement detected)")
                SetNekoState(NekoState.AWAKE)
            elif pet.NekoStateCount < pet.Duartions[pet.state]:
                dbg(f"akubi: waiting, count={pet.NekoStateCount}/{pet.Duartions[pet.state]}")
                pass
            else:
                dbg(f"akubi -> sleep (time expired)")
                SetNekoState(NekoState.SLEEP)

        case NekoState.SLEEP:
            if IsNekoMoveStart():
                dbg(f"sleep -> awake (cursor movement detected)")
                SetNekoState(NekoState.AWAKE)
            # Continue sleeping until time expires or movement starts

        case NekoState.AWAKE:
            if pet.NekoStateCount < pet.Duartions[pet.state]:
                dbg(f"awake: waiting, count={pet.NekoStateCount}/{pet.Duartions[pet.state]}")
                pass
            else:
                dbg(f"awake -> NekoDirection() (time expired)")
                NekoDirection()

        case _ if pet.state in MOVE_STATES:
            # Movement states
            pet.NekoX += pet.NekoMoveDx
            pet.NekoY += pet.NekoMoveDy
            dbg(f"movement: pos=({pet.NekoX},{pet.NekoY}), dx={pet.NekoMoveDx}, dy={pet.NekoMoveDy}")
            NekoDirection()
            if IsWindowOver():
                if IsNekoDontMove():
                    dbg(f"movement -> stop (screen boundary hit and neko hasn't moved)")
                    SetNekoState(NekoState.STOP)

        case _ if pet.state in SCRATCH_STATES:
            if IsNekoMoveStart():
                dbg(f"togi -> awake (cursor movement detected)")
                SetNekoState(NekoState.AWAKE)
            elif pet.NekoStateCount < pet.Duartions[pet.state]:
                dbg(f"togi: waiting, count={pet.NekoStateCount}/{pet.Duartions[pet.state]}")
                pass
            else:
                dbg(f"togi -> kaki (time expired)")
                SetNekoState(NekoState.KAKI)

        case _:
            # Default fallback
            dbg(f"unknown state '{pet.state}' -> stop")
            SetNekoState(NekoState.STOP)


    Interval()
   
#===================================================================================

# ---- Main process adapted ----
def ProcessNeko():
    global pet, screen, global_display

    # Initialize position
    pet.NekoX = (screen.WindowWidth - pet.bitmap_width) // 2
    pet.NekoY = (screen.WindowHeight - pet.bitmap_height) // 2
    pet.NekoLastX = pet.NekoX
    pet.NekoLastY = pet.NekoY
    SetNekoState(NekoState.STOP)

    while (ProcessEvent()):
        NekoThinkDraw()

# ---- Main entry ----
def main():

    GetArguments()
    InitScreen()
    LoadPlugins()    

    ProcessNeko()     

    StopPlugins()   
    FreeScreen()

if __name__=="__main__":
    main()