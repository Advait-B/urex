# UREx: take a photo from the NUCLEO-C542RC

You press the blue **B1** button on the Nucleo. The OpenMV camera takes a photo. The photo comes back to the laptop as a `.jpg` file.

```
 [B1 button]                                            [Laptop]
     |                                                     ^
     v              UART (115200)            USB cable     |  pc/get_photo.py
 NUCLEO-C542RC  ---- PHOTO -------->  OpenMV Cam H7 Plus   |  saves photos/*.jpg
 (stm32/*.c)    <--- JPEG, 512 B at a time --  (openmv/*.py) 
     |                                                     |
     +---- passes each piece on over the ST-LINK USB ------+
```

| Folder | Runs on | Files |
|---|---|---|
| `stm32/` | NUCLEO-C542RC (built in STM32CubeIDE) | `main.c`, `bsp.h`, `bsp_c5.c`, `proto.h`, `proto.c` |
| `openmv/` | OpenMV Cam H7 Plus | `main.py`, `link.py` |
| `pc/` | Your laptop | `get_photo.py` |

---

## 0. Read this first: three things that will trip you up

**1. The STM32 doesn't touch the camera sensor itself. The OpenMV board does.**
The STM32C542 has no camera interface and only 64 KB of RAM. One VGA greyscale frame is 300 KB. So the OpenMV board (its own STM32H743 plus the OV5640 sensor) takes the picture and compresses it to JPEG. The Nucleo sends it commands over UART. It then pulls the JPEG back in 512-byte pieces and passes each piece to the laptop. The Nucleo never holds the whole photo. This matches the architecture in UREX-CMOS-002 Rev B.

**2. STM32CubeIDE 2.x no longer contains CubeMX, and the STM32C5 uses new tools.**
Since CubeIDE 2.0.0, STM32CubeMX is a separate program. The STM32C5 family is configured with **STM32CubeMX2**, which is not the old CubeMX. The generated code uses **HAL2**, whose function names differ from the HAL you see in older tutorials. For example, `mx_system_init()` replaces `HAL_Init()` + `SystemClock_Config()`, and `HAL_OK` is not 0. The workflow is: configure in CubeMX2 → generate a CMake project → import it into CubeIDE.

**3. Update CubeIDE 2.1.0 to 2.1.1 or newer.**
ST's own guide to importing CubeMX2 projects says CubeIDE **2.1.1** is the minimum. Older versions stop with a CMake error in `cmake/components.cmake` ("Expression did not evaluate to a known generator expression"). Use *Help → Check for Updates*, or install the latest version from st.com.

---

## 1. Install the software (one time, on a fresh computer)

| Tool | What it's for | Where |
|---|---|---|
| **STM32CubeIDE 2.1.1 or newer** | Build, flash, debug the Nucleo | st.com → STM32CubeIDE |
| **STM32CubeMX2** (1.0.x) | Pick pins/peripherals, generate the project | st.com → STM32CubeMX2 |
| **OpenMV IDE** (latest) | Update the camera firmware and copy files to it | openmv.io/pages/download |
| **Python 3** + `pip install pyserial` | Run `get_photo.py` on the laptop | python.org |

You need an ST account to download from st.com. On first launch, CubeMX2 downloads the STM32C5 software pack automatically.

---

## 2. Prepare the Nucleo board

1. **Power jumper JP1.** There are two ways to power the board:
   - **JP1 on 1-2 (factory default):** the ST-LINK USB port (CN1) powers the board. **One cable** into CN1 then powers, programs and talks to the laptop. This is the simplest setup on the bench.
   - **JP1 on 7-8** (as in your *Mega Datasheet* notes): the second USB-C port (CN13) powers the board. You then need **two cables**: CN13 for power and CN1 for programming and the serial link to the laptop.
   
   Pick one. UM3615 says to select only one 5 V source.
2. **Solder bridges SB48 and SB49 must be closed (ON).** Without them, Arduino pins D0/D1 are not connected to PB15/PB14 and the camera link is dead. Look at the bottom of the board, or check UM3615.
3. LD1 (the green LED on PA5) needs SB8 ON. That is the factory state.

---

## 3. Prepare the camera (OpenMV IDE)

Do this with **only the OpenMV board** on USB and the Nucleo **not** wired to it.

1. Open OpenMV IDE and click **Connect** (bottom left). If it offers a firmware update, accept it. `main.py` uses the newer `csi` module, and it falls back to the old `sensor` module on older firmware.
2. Run the default "hello world" script. You should see live video. If you don't, reseat the camera module first.
3. Open the camera's USB drive (it appears like a memory stick). Copy **`openmv/link.py`** and **`openmv/main.py`** onto it, replacing the existing `main.py`.
4. *(Optional)* Put a microSD card in the camera. Every photo will then also be saved on it as `photo_0000.jpg`, `photo_0001.jpg`, …
5. Eject the drive and **unplug the camera's USB cable**. From now on the Nucleo powers it.

To change colour or resolution, edit the top of `main.py`. `PIXFORMAT = "RGB565"` gives colour. `FRAMESIZE = "VGA"` gives 640×480, but it is about 4× slower to transfer.

---

## 4. Wire the two boards (4 wires)

| Wire | Nucleo pin | OpenMV pin | Why |
|---|---|---|---|
| Camera → Nucleo data | **CN9-1, D0 (PB15, USART1_RX)** | **P4** (UART3 TX) | camera talks |
| Nucleo → camera data | **CN9-2, D1 (PB14, USART1_TX)** | **P5** (UART3 RX) | Nucleo talks |
| Ground | CN6-6 **GND** | **GND** | common reference, required |
| Power | CN6-5 **+5V** | **VIN** | Nucleo powers the camera |

- **TX goes to RX.** The pin that sends on one board goes to the pin that listens on the other.
- **Never connect the two 3V3 pins together**, and never plug the camera's USB in while VIN is wired.
- Both boards use 3.3 V logic, so no level shifter is needed.

---

## 5. Create the STM32 project in STM32CubeMX2

1. **Home → Board**. Search **NUCLEO-C542RC** and click **Continue**. Enter a project name, e.g. `urex_camera`, choose a folder, and click **Automatically Download, Install & Create Project**.
2. **Peripherals → Connectivity → USART1**: set the mode to **Asynchronous**.
   - 115200 baud, 8 data bits, no parity, 1 stop bit, no flow control.
   - Direction **TX and RX**.
   - Scroll to the GPIO settings and make sure **TX = PB14** and **RX = PB15**. Change them if the defaults differ (they must match the wires above).
   - If there is an **Overrun** option, set it to **Disable**. The code also does this itself.
3. **Peripherals → Connectivity → USART2**: **Asynchronous**, 115200 8N1, **TX = PA2** (RX = PA3). This is the ST-LINK virtual COM port, the link to the laptop. Check the pins: ST's own UART guide warns that PA2/PA3 may not be the defaults.
4. You don't need to set up the button (PC13) or the LED (PA5). `bsp_c5.c` configures them itself.
5. **Project Settings → IDE project generation**: set **Format = CMake** and **Toolchain = GCC**. The default toolchain is AC6 (Arm Compiler), which CubeIDE doesn't use, so you must change this.
6. Click **Generate**.

---

## 6. Add the code

1. In the generated project, find the folder that contains `main.c` (in ST's projects this is `application/`).
2. Copy all 5 files from `stm32/` into that folder. Let our `main.c` **replace** the generated one.
3. Open the **top-level `CMakeLists.txt`** and add the two new `.c` files where it says *Add additional source files here*. To get the path right, open `cmake/files.cmake` and see how it writes the path to `main.c`. Use the same path, with `bsp_c5.c` and `proto.c` in place of `main.c`. In ST's layout it looks like this:

```cmake
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
  # Add additional source files here
  ../../application/bsp_c5.c
  ../../application/proto.c
)
```

> Why this step: CubeMX2 only lists the files it generated. Any `.c` file you add yourself has to be listed here, or the linker reports "undefined reference to `bsp_init`".

If you later change pins in CubeMX2 and regenerate, CubeMX2 keeps its generated code in separate files, so your files survive. Check afterwards that `main.c` is still ours.

---

## 7. Import, build and flash in STM32CubeIDE

1. Start STM32CubeIDE and pick a workspace folder.
2. **File → New → STM32 Project → STM32 Project Create/Import**, then choose **STM32CubeMX2 project**.
3. On *Project with existing CMake sources*, enter a name. **Browse** to the folder you generated in step 5 (the one containing `CMakeLists.txt`).
4. Set Toolchain = **MCU ARM GCC** and MCU = **STM32C542RCTx** (the same chip as in CubeMX2). Click **Finish**.
5. Right-click the project → **Build Project**. It should end with no errors.
6. Plug the Nucleo's **CN1** (ST-LINK USB) into the laptop. Choose **Run → Debug As → STM32 C/C++ Application** and accept the default ST-LINK settings. If CubeIDE offers an ST-LINK firmware upgrade, accept it. When the debugger stops at `main`, press **Resume (F8)**, or use **Run As** to just flash and run.

---

## 8. Take a photo

1. Find the Nucleo's serial port:
   - **Windows:** Device Manager → Ports → "STMicroelectronics STLink Virtual COM Port (COMx)"
   - **macOS:** `ls /dev/tty.usbmodem*`
   - **Linux:** usually `/dev/ttyACM0`
2. Close any serial terminal that has this port open. Only one program can use a port at a time.
3. Run:
   ```
   python pc/get_photo.py COM5
   ```
4. Press the **reset** button (black) on the Nucleo. You should see `camera online`.
5. Press **B1** (blue). The green LED stays on while the photo transfers. Then you see:
   ```
   receiving 320x240 photo, 11234 bytes ......................
   saved photos/photo_20261008_171404.jpg  [OK]
   ```
   `[OK]` means the file passed a CRC-32 check. Every byte that reached the laptop matches what the Nucleo received from the camera.

At 115200 baud, a typical 320×240 greyscale photo (about 8–15 KB) takes roughly 1.5–3 s. A busy, high-detail image can be 30 KB or more and take about 6 s. To go faster, raise the baud rate in all four places together: USART1 and USART2 in CubeMX2, `Link(baud=…)` in `main.py`, and `--baud` for `get_photo.py`.

---

## 9. If it doesn't work

| What you see | Most likely cause | Fix |
|---|---|---|
| CMake error in `components.cmake` during import | CubeIDE older than 2.1.1 | Update CubeIDE |
| Build error: `undefined reference to bsp_init` / `proto_send` | `.c` files not listed in CMakeLists | Step 6.3 |
| Build error: `mx_usart1.h: No such file` | USART1 not enabled in CubeMX2 | Step 5.2, then Generate again |
| Build fails with odd compiler flags | Toolchain left on AC6 | Step 5.5: set GCC |
| `camera not answering` | TX/RX swapped, SB48/SB49 open, no GND wire, camera USB still plugged in, `main.py` not on the camera | Check step 4. In OpenMV IDE, run `main.py` and confirm it doesn't crash |
| Nothing at all in `get_photo.py` | Wrong port, or another terminal has it open, or USART2 not on PA2 | Step 8.1 / 5.3 |
| `FAIL lost a piece of the photo` | Noisy or long wires | Keep the wires under 20 cm, twist each data wire with GND |
| Three quick LED blinks | Any `FAIL` | Read the message printed by `get_photo.py` |
| Photo all black or white | Exposure still settling, or lens cap / focus | Wait 2 s after power-up. Focus the M12 lens in OpenMV IDE first |

---

## 10. How it works (for the report)

**Link frame** (unchanged from UREX-CMOS-002 Rev B §3):
`0xA5 | CMD | LEN (u16, little-endian) | PAYLOAD | CRC-8 (poly 0x07, init 0)`. The response is `CMD | 0x80`.

**Two new commands** (PING/SNAP/GRAB/PROFILE/EXPOSE are unchanged, so later stages still work):

| CMD | Request | Response |
|---|---|---|
| `0x06 PHOTO` | u8 JPEG quality (1–100) | `0x86` u32 size, u16 width, u16 height, u16 SD file number (`0xFFFF` = not saved) |
| `0x07 READ` | u32 offset, u16 n (≤ 512) | `0x87` u32 offset, n bytes of the JPEG |

**Why the photo moves in pieces:** a whole JPEG may not fit in the Nucleo's 64 KB RAM, and later stages need that RAM. Each `READ` asks for one piece by its offset, so a piece that arrives with a CRC error can simply be asked for again (up to 3 tries). The Nucleo only needs a 516-byte buffer.

**Why overrun detection is switched off:** HAL2's polling receive (`HAL_UART_Receive`) never clears a UART *overrun* error. If one stray byte arrived while the Nucleo wasn't listening, an overrun could block reception from then on. With detection off, a stray byte is simply dropped. The CRC and the retry then handle anything that goes wrong.

**What was checked before handing this over:**
- The firmware was cross-compiled with ARM GCC inside ST's own NUCLEO-C542RC example project (STM32CubeC5 2.0.0, HAL2). It built with zero warnings using `-Wall`: about 21 KB flash and 2.7 KB RAM in the debug build.
- The full chain ran in a simulation: the real `main.c` and `proto.c` (built for the PC), the real `openmv/main.py` and `link.py` (with a fake camera), and the real `get_photo.py`. Both photos were saved, CRC OK, and a deliberately corrupted piece was recovered by the retry.
- **Not yet tested on the real boards.** The CubeMX2 menu names above come from ST's guides, and the exact labels may differ slightly in your version.

References: UM3615 (NUCLEO-C542RC user manual) · STM32C542RC datasheet · UREX-CMOS-002 Rev B · ST Community, "How to import a STM32CubeMX2 CMake project into STM32CubeIDE" and "How to use UART with STM32CubeMX2" · STM32CubeC5 on GitHub (HAL2 source and C542RC examples) · OpenMV `csi` and `image` docs.
