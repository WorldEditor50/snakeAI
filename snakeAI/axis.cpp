#include "axis.h"

Axis::Axis(QWidget *parent) :
    QWidget(parent),
    interval(40),
    scale(1)
{
    setMinimumSize(QSize(600, 600));
    QPalette pal;
    pal.setBrush(backgroundRole(), Qt::white);
    x = 0;
}

void Axis::addPoint(double y)
{
    QPointF p(x, y);
    points.append(p);
    x++;
    update();
    return;
}

void Axis::setInterval(int value)
{
    interval = value;
    update();
    return;
}

void Axis::setScale(int value)
{
    scale = value;
    update();
    return;
}

void Axis::paintEvent(QPaintEvent *ev)
{
    Q_UNUSED(ev)
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
    painter.drawText(-w + 20, -h + 20, QString("SCALE:x%1").arg(scale));
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
         painter.drawText(-40, i, QString("%1").arg(-i));
    }
    for (int i = interval; i < h; i += interval) {
        painter.drawText(-40, i, QString("%1").arg(-i));
    }
    /* curve */
    pen.setColor(QColor(0, 150, 250));
    pen.setWidthF(1);
    painter.setPen(pen);
    for (int i = 1; i < points.size(); i++) {
        qreal x1 = points.at(i - 1).x();
        qreal y1 = points.at(i - 1).y();
        qreal x2 = points.at(i).x();
        qreal y2 = points.at(i).y();
        painter.drawLine(x1, -y1, x2, -y2);
        //points.replace(i - 1, points.at(i));
    }
    for (auto it = points.begin(); it != points.end();it++) {
        qreal x = it->x();
        if (x < -w) {
           it = points.erase(it);
        } else {
           it->setX(x - 1);
        }
    }
    return;
}

void Axis::timerEvent(QTimerEvent *event)
{
    if (event->timerId() == timerID) {
        double y = rand() % 200 - rand() % 200;
        QPointF p(x, y);
        points.append(p);
        x++;
        update();
    }
    return;
}

