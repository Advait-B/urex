/* bsp_c5.c - NUCLEO-C542RC board support. The ONLY file that calls HAL2.
 *
 * Expects an STM32CubeMX2 project with:
 *   USART1  Asynchronous, 115200 8N1, TX on PB14, RX on PB15   (camera)
 *   USART2  Asynchronous, 115200 8N1, TX on PA2 (RX PA3 opt.) (ST-LINK VCP)
 * CubeMX2 generates mx_usart1.c/.h and mx_usart2.c/.h for these.
 *
 * API names checked against STM32CubeC5 2.0.0 (stm32c5xx_hal_uart.h,
 * stm32c5xx_hal_gpio.h) and the NUCLEO-C542RC example projects.
 */
#include "bsp.h"
#include <string.h>
#include "mx_system.h"   /* mx_hal_def.h -> stm32_hal.h (HAL2 headers) */
#include "mx_usart1.h"
#include "mx_usart2.h"

#define LED_PORT    HAL_GPIOA
#define LED_PIN     HAL_GPIO_PIN_5      /* LD1 green, via SB8 */
#define BJT_PORT    HAL_GPIOA
#define BJT_PIN     HAL_GPIO_PIN_10     /* Arduino D2 (CN9-3) -> BJT base resistor */
#define BTN_PORT    HAL_GPIOC
#define BTN_PIN     HAL_GPIO_PIN_13     /* B1 blue, high when pressed */

static hal_uart_handle_t *cam;          /* USART1 */
static hal_uart_handle_t *vcp;          /* USART2 */

/* CubeMX2 may or may not call mx_usartN_uart_init() inside mx_system_init()
 * (it depends on the "generate and call initialization" option), so call it
 * here only if the handle is still unconfigured. */
static hal_uart_handle_t *uart_up(hal_uart_handle_t *h, hal_uart_handle_t *(*init)(void))
{
    if (HAL_UART_GetState(h) == HAL_UART_STATE_RESET)
        return init();
    return h;
}

int bsp_init(void)
{
    cam = uart_up(mx_usart1_uart_gethandle(), mx_usart1_uart_init);
    vcp = uart_up(mx_usart2_uart_gethandle(), mx_usart2_uart_init);
    if (cam == NULL || vcp == NULL)
        return -1;

    /* Polling receive does not clear an overrun error. If a stray byte ever
     * arrives while we are not listening, an enabled overrun detector can
     * stall reception, so turn it off (stray bytes are simply dropped). */
    (void)HAL_UART_DisableRxOverRunDetection(cam);

    /* LED and button: set them up here so the CubeMX2 pinout only needs the
     * two UARTs. Re-initialising pins CubeMX2 already set up is harmless. */
    hal_gpio_config_t g;
    memset(&g, 0, sizeof g);

    HAL_RCC_GPIOA_EnableClock();
    g.mode        = HAL_GPIO_MODE_OUTPUT;
    g.speed       = HAL_GPIO_SPEED_FREQ_LOW;
    g.pull        = HAL_GPIO_PULL_NO;
    g.output_type = HAL_GPIO_OUTPUT_PUSHPULL;
    g.init_state  = HAL_GPIO_PIN_RESET;
    if (HAL_GPIO_Init(LED_PORT, LED_PIN, &g) != HAL_OK)
        return -2;
    if (HAL_GPIO_Init(BJT_PORT, BJT_PIN, &g) != HAL_OK)   /* same settings, starts LOW = BJT off */
        return -4;

    HAL_RCC_GPIOC_EnableClock();
    g.mode = HAL_GPIO_MODE_INPUT;       /* board has its own pull-down on B1 */
    g.pull = HAL_GPIO_PULL_NO;
    if (HAL_GPIO_Init(BTN_PORT, BTN_PIN, &g) != HAL_OK)
        return -3;

    return 0;
}

int bsp_cam_tx(const void *p, uint32_t n)
{
    if (n == 0) return 0;
    return HAL_UART_Transmit(cam, p, n, 200) != HAL_OK;
}

int bsp_cam_rx(void *p, uint32_t n, uint32_t timeout_ms)
{
    if (n == 0) return 0;
    return HAL_UART_Receive(cam, p, n, timeout_ms) != HAL_OK;
}

void bsp_cam_flush(void)
{
    uint8_t b;
    /* timeout 0: returns at once when no byte is waiting */
    for (int i = 0; i < 4096 && HAL_UART_Receive(cam, &b, 1, 0) == HAL_OK; i++) { }
}

void bsp_vcp_write(const void *p, uint32_t n)
{
    if (n) (void)HAL_UART_Transmit(vcp, p, n, 1000);
}

void bsp_vcp_print(const char *s)
{
    bsp_vcp_write(s, (uint32_t)strlen(s));
}

int bsp_button(void)
{
    return HAL_GPIO_ReadPin(BTN_PORT, BTN_PIN) == HAL_GPIO_PIN_SET;
}

void bsp_led(int on)
{
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, on ? HAL_GPIO_PIN_SET : HAL_GPIO_PIN_RESET);
}

void bsp_bjt(int on)
{
    HAL_GPIO_WritePin(BJT_PORT, BJT_PIN, on ? HAL_GPIO_PIN_SET : HAL_GPIO_PIN_RESET);
}

void     bsp_delay_ms(uint32_t ms) { HAL_Delay(ms); }
uint32_t bsp_millis(void)          { return HAL_GetTick(); }
