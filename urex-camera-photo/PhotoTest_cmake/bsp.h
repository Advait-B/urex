/* bsp.h - what the application needs from the NUCLEO-C542RC board.
 *
 * Only bsp_c5.c calls the STM32C5 HAL2 drivers. Everything else (proto.c,
 * main.c) uses these functions, so the CCD branch or a future board only has
 * to replace bsp_c5.c.
 *
 * UREx CMOS branch - see UREX-CMOS-002 Rev B, Stage 1.
 */
#ifndef BSP_H
#define BSP_H

#include <stdint.h>

/* Board start-up: UART handles, overrun handling, button and LED pins.
 * Call once, after mx_system_init(). Returns 0 when everything is ready. */
int      bsp_init(void);

/* Camera link: USART1, PB14 = TX (Arduino D1), PB15 = RX (Arduino D0). */
int      bsp_cam_tx(const void *p, uint32_t n);                 /* 0 = ok */
int      bsp_cam_rx(void *p, uint32_t n, uint32_t timeout_ms);  /* 0 = ok */
void     bsp_cam_flush(void);                                   /* drop stale RX bytes */

/* Laptop link: USART2 (PA2/PA3) -> ST-LINK virtual COM port on the USB cable. */
void     bsp_vcp_write(const void *p, uint32_t n);
void     bsp_vcp_print(const char *s);

int      bsp_button(void);          /* B1 (blue, PC13): 1 = pressed */
void     bsp_led(int on);           /* LD1 (green, PA5)             */
void     bsp_delay_ms(uint32_t ms);
uint32_t bsp_millis(void);

#endif /* BSP_H */
