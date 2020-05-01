#include "mainwindow.h"
#include <windows.h>
#include <DbgHelp.h>
#include <QApplication>

LONG ApplicationCrashHandler(EXCEPTION_POINTERS *pException)
{
    /* create dump file */
    HANDLE hDumpFile = CreateFile(L"crash.dmp",
                                  GENERIC_WRITE,
                                  0,
                                  NULL,
                                  CREATE_ALWAYS,
                                  FILE_ATTRIBUTE_NORMAL,
                                  NULL);
    if (hDumpFile != INVALID_HANDLE_VALUE) {
        /* dump info */
        MINIDUMP_EXCEPTION_INFORMATION dumpInfo;
        dumpInfo.ExceptionPointers = pException;
        dumpInfo.ThreadId = GetCurrentThreadId();
        dumpInfo.ClientPointers = TRUE;
        /* write dump info */
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hDumpFile, MiniDumpNormal, &dumpInfo, NULL, NULL);
    }
#if 0
    MsgBox *msgbox = new MsgBox;
    msgbox->setInfo(QObject::tr("Program crash"), QObject::tr("Express my sincere apologies for the mistake!"), QPixmap(":/msgbox/attention.png"), true, true);
    msgbox->exec();
#endif
    return EXCEPTION_EXECUTE_HANDLER;
}
int main(int argc, char *argv[])
{
    SetUnhandledExceptionFilter((LPTOP_LEVEL_EXCEPTION_FILTER)ApplicationCrashHandler);//注冊异常捕获函数
    srand((unsigned int)time(nullptr));
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
