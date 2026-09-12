package com.example.microvlmptq

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * Synthetic scene generator, pixel-for-pixel parity with python/synthetic.py.
 * Renders on a 64x64 bitmap; RGB values are later /255 just like training.
 */
class SceneGenerator {

    data class Scene(val bitmap: Bitmap, val caption: String)

    companion object {
        const val IMG_SIZE = 64
        private val SHAPES = arrayOf("circle", "square", "triangle", "star", "diamond")
        private val COLORS = arrayOf("red", "green", "blue", "yellow", "magenta", "cyan", "orange")
        private val SIZES = arrayOf("small", "large")
        private val POSITIONS = arrayOf("left", "center", "right")

        private val COLOR_RGB = mapOf(
            "red" to intArrayOf(220, 60, 60),
            "green" to intArrayOf(60, 200, 90),
            "blue" to intArrayOf(70, 120, 235),
            "yellow" to intArrayOf(235, 220, 70),
            "magenta" to intArrayOf(220, 70, 190),
            "cyan" to intArrayOf(70, 200, 220),
            "orange" to intArrayOf(240, 150, 50),
        )
        private const val BG = -0xFFE6E2ED.toInt() // (18,18,30) placeholder
        private val BG_RGB = intArrayOf(18, 18, 30)
        private val POS_CENTER = mapOf(
            "left" to intArrayOf(16, 32),
            "center" to intArrayOf(32, 32),
            "right" to intArrayOf(48, 32),
        )

        fun captionOf(shape: String, color: String, size: String, pos: String): String =
            "a $size $color $shape at the $pos"

        fun randomScene(ctx: Context, rng: Random = Random.Default): Scene {
            val shape = SHAPES[rng.nextInt(SHAPES.size)]
            val color = COLORS[rng.nextInt(COLORS.size)]
            val size = SIZES[rng.nextInt(SIZES.size)]
            val pos = POSITIONS[rng.nextInt(POSITIONS.size)]
            val bmp = render(shape, color, size, pos)
            return Scene(bmp, captionOf(shape, color, size, pos))
        }

        fun render(shape: String, colorName: String, size: String, pos: String): Bitmap {
            val bmp = Bitmap.createBitmap(IMG_SIZE, IMG_SIZE, Bitmap.Config.ARGB_8888)
            val c = Canvas(bmp)
            c.drawColor(Color.rgb(BG_RGB[0], BG_RGB[1], BG_RGB[2]))

            val cx = POS_CENTER.getValue(pos)[0]
            val cy = POS_CENTER.getValue(pos)[1]
            val rgb = COLOR_RGB.getValue(colorName)
            val r = if (size == "large") 16f else 9f
            val paint = Paint()
            paint.isAntiAlias = true
            paint.color = Color.rgb(rgb[0], rgb[1], rgb[2])

            when (shape) {
                "circle" -> c.drawCircle(cx.toFloat(), cy.toFloat(), r, paint)
                "square" -> c.drawRect(cx - r, cy - r, cx + r, cy + r, paint)
                "triangle" -> {
                    val p = Path().apply {
                        moveTo(cx.toFloat(), cy - r)
                        lineTo(cx - r, cy + r)
                        lineTo(cx + r, cy + r)
                        close()
                    }
                    c.drawPath(p, paint)
                }
                "star" -> {
                    val p = Path()
                    for (i in 0 until 10) {
                        val ang = Math.toRadians((-90.0 + 36.0 * i))
                        val rad = if (i % 2 == 0) r else r * 0.45f
                        val x = (cx + rad * cos(ang)).toFloat()
                        val y = (cy + rad * sin(ang)).toFloat()
                        if (i == 0) p.moveTo(x, y) else p.lineTo(x, y)
                    }
                    p.close()
                    c.drawPath(p, paint)
                }
                "diamond" -> {
                    val p = Path().apply {
                        moveTo(cx.toFloat(), cy - r)
                        lineTo(cx + r, cy.toFloat())
                        lineTo(cx.toFloat(), cy + r)
                        lineTo(cx - r, cy.toFloat())
                        close()
                    }
                    c.drawPath(p, paint)
                }
            }
            return bmp
        }

        /** Bitmap -> float[64*64*3] in HWC order, values /255 (training parity). */
        fun toFloatInput(bmp: Bitmap, scale: Boolean = true): FloatArray {
            val out = FloatArray(IMG_SIZE * IMG_SIZE * 3)
            val px = IntArray(IMG_SIZE * IMG_SIZE)
            bmp.getPixels(px, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)
            for (i in px.indices) {
                val p = px[i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                val base = i * 3
                if (scale) {
                    out[base] = r / 255f
                    out[base + 1] = g / 255f
                    out[base + 2] = b / 255f
                } else {
                    out[base] = r.toFloat()
                    out[base + 1] = g.toFloat()
                    out[base + 2] = b.toFloat()
                }
            }
            return out
        }
    }
}