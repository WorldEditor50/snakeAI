#ifndef AXIS_H
#define AXIS_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QWheelEvent>
#include <QTimer>
#include <cmath>
#include <mutex>

class AxisWidget : public QWidget
{
    Q_OBJECT
public:
    explicit AxisWidget(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;
    void timerEvent(QTimerEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
signals:

public slots:
    void addPoint(float y);
    void setInterval(int value);
    void setScale(int value);
    void clearPoints();
public:
    std::mutex mutex;
    QList<QPointF> points;
    int timerID;
    float x = 0;
    int interval;
    int scale;
};

#endif // AXIS_H
