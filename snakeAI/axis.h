#ifndef AXIS_H
#define AXIS_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QWheelEvent>
#include <QTimer>
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
private:
    /* Scrolls the existing samples one step left and appends `y` at the right
       edge. Shared by addPoint() and timerEvent(). */
    void appendSample(float y);
public:
    std::mutex mutex;
    QList<QPointF> points;
    int timerID;
    /* Total samples received since the last clearPoints(). Used to be the
       sample's DRAWING coordinate, which is what made the curve walk off the
       right edge of the widget. */
    float x = 0;
    int interval;
    int scale;
};

#endif // AXIS_H
