/* Audit successful clock opens without changing the files or returned handles. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

static void record_clock(int fd, const char *path) {
    if(fd<0 || !path || (!strstr(path,".clk") && !strstr(path,"/clock/"))) return;
    const char *out=getenv("CLOCK_TRACE");
    if(!out) return;
    char link[64], resolved[8192], line[8300];
    snprintf(link,sizeof(link),"/proc/self/fd/%d",fd);
    ssize_t n=readlink(link,resolved,sizeof(resolved)-1);
    if(n<0) return;
    resolved[n]=0;
    int (*real_open)(const char*,int,...)=dlsym(RTLD_NEXT,"open");
    int log=real_open(out,O_WRONLY|O_APPEND|O_CREAT,0600);
    if(log<0) return;
    int bytes=snprintf(line,sizeof(line),"%ld\t%s\n",(long)getpid(),resolved);
    write(log,line,bytes);close(log);
}
FILE *fopen(const char *path,const char *mode) {
    FILE *(*real)(const char*,const char*)=dlsym(RTLD_NEXT,"fopen");
    FILE *f=real(path,mode);if(f)record_clock(fileno(f),path);return f;
}
FILE *fopen64(const char *path,const char *mode) {
    FILE *(*real)(const char*,const char*)=dlsym(RTLD_NEXT,"fopen64");
    FILE *f=real(path,mode);if(f)record_clock(fileno(f),path);return f;
}
