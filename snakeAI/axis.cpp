#include "axis.h"

AxisWidget::AxisWidget(QWidget *parent) :
    QWidget(parent),
    interval(40),
    scale(8)
{
    setMinimumSize(QSize(600, 600));
    QPalette pal;
    pal.setBrush(backgroundRole(), Qt::white);
    x = 0;
    timerID = -1;   /* timerEvent() compares against this; it was left uninitialised */
}

void AxisWidget::appendSample(float y)
{
    /*
       Scroll one step and place the new sample at the RIGHT edge.

       The scrolling used to live in paintEvent(). That is not a time base:
       paintEvent also runs on resize/expose/occlusion, and Qt coalesces the
       update() calls that arrive from addPoint() and from the widget's
       readyForPaint signal, so the number of shifts per sample was arbitrary.

       Worse, a sample's x came from a counter that only ever increased
       (`x++` in addPoint), while the shift was per repaint. Once that counter
       passed width()/2 every new sample was placed outside the visible window
       [-w, +w] and the curve stopped updating — and because clearPoints() did
       NOT reset the counter, switching agent (which emits clearReward) left the
       reward window permanently blank: new samples kept arriving at x = 4000,
       4001, ... with the window showing only ~±300.

       Tying the scroll to the arrival of a sample makes the horizontal axis
       mean "how many samples ago", which is what the plot is for.
    */
    const qreal right = width() / 2.0;
    for (int i = 0; i < points.size(); i++) {
        points[i].setX(points[i].x() - 1);
    }
    while (!points.isEmpty() && points.first().x() < -right) {
        points.removeFirst();
    }
    points.append(QPointF(right, y));
    x += 1;
    update();
    return;
}

void AxisWidget::addPoint(float y)
{
    appendSample(y);
    return;
}

void AxisWidget::setInterval(int value)
{
    interval = value;
    update();
    return;
}

void AxisWidget::setScale(int value)
{
    scale = value;
    update();
    return;
}

void AxisWidget::clearPoints()
{
    /*
       Nothing about a sample's position depends on state that outlives the
       clear any more (see appendSample()), so clearing is enough. This used to
       leave the x counter untouched, which is why switching agent — the one
       thing that calls clearReward() — left the reward window unable to show
       any further data.
    */
    points.clear();
    x = 0;
    update();
    return;
}

void AxisWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)
    int w = this->width() / 2;
    int h = this->height() / 2;
    QPainter painter(this);
    QPen pen;
    pen.setStyle(Qt::SolidLine);
    pen.setWidthF(0.5);
    painter.setPen(pen);
    /* convert Qt-axis to Descartes-axis */
    painter.setViewport(0, 0, 2 * w, 2 * h);
    painter.setWindow(-w, -h, 2 * w, 2 * h);
    painter.fillRect(-w, -h, 2 * w,  2 * h, Qt::black);
    /* x, y-axis */
    pen.setWidthF(1);
    pen.setColor(Qt::white);
    painter.setPen(pen);
    painter.drawLine(-w, 0, w, 0);
    painter.drawLine(0, h, 0, -h);
    /* draw scale */
    painter.drawText(-w + 20, -h + 20, QString("Y-SCALE:x%1").arg(scale));
    /* grid */
    pen.setWidthF(0.3);
    pen.setColor(Qt::gray);
    painter.setPen(pen);
    for (int i = -w; i < w; i++) {
        if (i % 20 == 0) {
            painter.drawLine(i, -h, i, h);
        }
    }
    for (int i = -h; i < h; i++) {
        if (i % 20 == 0) {
            painter.drawLine(-w, i, w, i);
        }
    }
    /* mark */
    pen.setColor(Qt::gray);
    painter.setPen(pen);

    for (int i = 0; i >= -w; i -= interval) {
        painter.drawText(i, 20, QString("%1").arg(i));
    }
    for (int i = 0; i < w; i += interval) {
        painter.drawText(i, 20, QString("%1").arg(i));
    }
    for (int i = -interval; i >= -h; i -= interval) {
         painter.drawText(-40, i, QString("%1").arg(-i / scale));
    }
    for (int i = interval; i < h; i += interval) {
        painter.drawText(-40, i, QString("%1").arg(-i / scale));
    }
    /* curve */
    pen.setColor(QColor(0, 150, 250));
    pen.setWidthF(1);
    painter.setPen(pen);
    for (int i = 1; i < points.size(); i++) {
        qreal x1 = points.at(i - 1).x();
        qreal y1 = points.at(i - 1).y() * scale;
        qreal x2 = points.at(i).x();
        qreal y2 = points.at(i).y() * scale;
        painter.drawLine(x1, -y1, x2, -y2);
        //points.replace(i - 1, points.at(i));
    }
    /*
       The scrolling/erasing loop that used to sit here MOVED to
       appendSample(). Doing it in paintEvent() made the x axis depend on how
       many times Qt repainted the widget rather than on how many samples
       arrived, and mutating the sample list from a paint handler is wrong in
       its own right (paintEvent must be a pure read of the model).
    */
    return;
}

void AxisWidget::timerEvent(QTimerEvent *event)
{
    if (event->timerId() == timerID) {
        float y = rand() % 200 - rand() % 200;
        appendSample(y);
    }
    return;
}

void AxisWidget::wheelEvent(QWheelEvent *event)
{

#if (QT_VERSION >= QT_VERSION_CHECK(6,0,0))
    int delta = event->angleDelta().y();
#else
    int delta = event->delta();
#endif
    if (delta < 0) {
        scale /= 2;
        scale = scale < 1 ? 1 : scale;
    } else {
        scale *= 2;
        scale = scale > 32 ? 32 : scale;
    }
    event->accept();
    return;
}

