INCLUDEPATH += $$PWD

SOURCES += $$PWD/bpnn.cpp \
            $$PWD/conv2d.cpp \
            $$PWD/ddpg.cpp \
            $$PWD/dpg.cpp \
            $$PWD/dqn.cpp \
            $$PWD/drpg.cpp \
            $$PWD/gru.cpp \
            $$PWD/lstm.cpp \
            $$PWD/ppo.cpp \
            $$PWD/qlstm.cpp \
            $$PWD/util.cpp

HEADERS += $$PWD/bpnn.h \
            $$PWD/activate.h \
            $$PWD/conv2d.h \
            $$PWD/ddpg.h \
            $$PWD/dpg.h \
            $$PWD/dqn.h \
            $$PWD/drpg.h \
            $$PWD/gru.h \
            $$PWD/layer.h \
            $$PWD/loss.h \
            $$PWD/lstm.h \
            $$PWD/lstmnet.h \
            $$PWD/mat.hpp \
            $$PWD/optimizer.h \
            $$PWD/ppo.h \
            $$PWD/qlstm.h \
            $$PWD/rl_basic.h \
            $$PWD/tensor.hpp \
            $$PWD/util.h
