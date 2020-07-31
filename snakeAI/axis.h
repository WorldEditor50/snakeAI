#ifndef AXIS_H
#define AXIS_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QTimer>
#include <cmath>

class Axis : public QWidget
{
    Q_OBJECT
public:
    explicit Axis(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *ev) override;
    void timerEvent(QTimerEvent *event) override;
signals:

public slots:
    void addPoint(double y);
public:
    QList<QPointF> points;
    int timerID;
    double x = 0;
};

#endif // AXIS_H
