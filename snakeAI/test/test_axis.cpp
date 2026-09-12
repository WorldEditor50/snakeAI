/*
 * test_axis.cpp — regression test for the reward plot (AxisWidget).
 *
 * The GUI layer had no test coverage at all, which is how this shipped:
 * switching the agent emptied the "Total reward/episode" window and it never
 * filled up again.
 *
 * Root cause (see appendSample() in axis.cpp):
 *   * a sample's x coordinate came from `x`, a counter that only ever
 *     increased, while the scrolling was done in paintEvent() — one shift per
 *     REPAINT, not per sample. Once the counter passed width()/2 the newest
 *     samples were placed outside the visible window [-w, +w];
 *   * clearPoints() (called on every agent switch via the clearReward signal)
 *     cleared the list but did NOT reset that counter, so after a switch every
 *     subsequent sample was drawn far off-screen and the plot looked dead.
 *
 * So the invariants worth pinning down are:
 *   1. the newest sample is always inside the visible window, however many
 *      samples have arrived;
 *   2. the retained sample list stays bounded;
 *   3. after clearPoints() the very next sample is immediately visible again;
 *   4. paintEvent() does not mutate the samples (painting is not a time base).
 *
 * paintEvent() is forced with QWidget::render(), which calls it regardless of
 * whether the widget is shown, so the test needs no window system.
 */
#include <QApplication>
#include <QPixmap>
#include <QList>
#include <QPointF>
#include <iostream>
#include <string>
#include <cmath>

#include "axis.h"

static int failures = 0;

static void check(bool ok, const std::string &what, const std::string &detail)
{
    std::cout << (ok ? "  PASS  " : "  FAIL  ") << what;
    if (!detail.empty()) {
        std::cout << "   [" << detail << "]";
    }
    std::cout << std::endl;
    if (!ok) {
        failures++;
    }
}

/* Force one real paintEvent(). */
static void forcePaint(AxisWidget &w)
{
    QPixmap pm(w.size());
    w.render(&pm);
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    std::cout << "=== Reward plot (AxisWidget) test ===" << std::endl;

    const int W = 600;
    const qreal right = W / 2.0;

    // ---------------- Test 1: samples stay visible and bounded ----------------
    std::cout << "\nTest 1: samples remain inside the visible window" << std::endl;
    {
        AxisWidget w;
        w.resize(W, W);

        const int N = 5000;                 // far more than width()/2
        for (int i = 0; i < N; i++) {
            w.addPoint(float(i % 100));
            forcePaint(w);
        }

        const qreal newest = w.points.last().x();
        const qreal oldest = w.points.first().x();

        std::cout << "  after " << N << " samples: size=" << w.points.size()
                  << " x=[" << oldest << ", " << newest << "]"
                  << " window=[" << -right << ", " << right << "]" << std::endl;

        check(newest <= right + 0.5,
              "newest sample is inside the window",
              "newest x=" + std::to_string(newest) + " must be <= " + std::to_string(right));
        check(oldest >= -right - 1.0,
              "oldest retained sample has not drifted off the left edge",
              "oldest x=" + std::to_string(oldest));
        check(int(w.points.size()) <= W + 2,
              "retained sample list is bounded by the widget width",
              "size=" + std::to_string(w.points.size()));
        check(int(w.points.size()) > W / 2,
              "the plot actually holds a history worth showing",
              "size=" + std::to_string(w.points.size()));
    }

    // ---------------- Test 2: clearPoints() then one sample ----------------
    std::cout << "\nTest 2: the first sample after clearPoints() is visible" << std::endl;
    {
        AxisWidget w;
        w.resize(W, W);

        for (int i = 0; i < 5000; i++) {
            w.addPoint(float(i % 100));
            forcePaint(w);
        }
        w.clearPoints();
        forcePaint(w);

        check(w.points.isEmpty(), "clearPoints() empties the plot",
              "size=" + std::to_string(w.points.size()));

        w.addPoint(42.0f);
        forcePaint(w);

        check(w.points.size() == 1, "exactly one sample after the clear",
              "size=" + std::to_string(w.points.size()));

        const qreal x = w.points.last().x();
        std::cout << "  first sample after clear: x=" << x
                  << " y=" << w.points.last().y() << std::endl;
        check(x >= -right && x <= right + 0.5,
              "first sample after clear is inside the window",
              "x=" + std::to_string(x));

        // This is the exact regression: switching agent clears the plot, so the
        // samples that follow the switch must be drawn where they can be seen.
        check(w.x == 1.0f, "the sample counter restarts with the clear",
              "x=" + std::to_string(w.x));
    }

    // ---------------- Test 3: paintEvent must not mutate the model ----------------
    std::cout << "\nTest 3: paintEvent() is a pure read of the samples" << std::endl;
    {
        AxisWidget w;
        w.resize(W, W);

        for (int i = 0; i < 50; i++) {
            w.addPoint(float(i));
        }
        const QList<QPointF> before = w.points;

        forcePaint(w);
        forcePaint(w);
        forcePaint(w);

        bool same = (before.size() == w.points.size());
        for (int i = 0; same && i < before.size(); i++) {
            same = (before.at(i) == w.points.at(i));
        }
        check(same, "repainting does not move or drop samples",
              "size " + std::to_string(before.size()) + " -> "
              + std::to_string(w.points.size()));
    }

    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Summary: " << (3 - (failures > 0 ? 1 : 0)) << "/3 tests passed"
              << (failures > 0 ? " (" + std::to_string(failures) + " checks FAILED)" : "")
              << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    return failures;
}
