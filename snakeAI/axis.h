#ifndef AXIS_H
#define AXIS_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QWheelEvent>
#include <QTimer>
#include <cmath>

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
    void addPoint(double y);
    void setInterval(int value);
    void setScale(int value);
public:
    QList<QPointF> points;
    int timerID;
    double x = 0;
    int interval;
    int scale;
};

#endif // AXIS_H
