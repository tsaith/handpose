# Make macros:
# $< = first input
# $^ = inputs
# $@ = outputs

CC  		= g++
#CC  		= clang++-6.0
CFLAGS		=
#CPPFLAGS    = -Iinclude/ -pthread -fpermissive -O3 -fPIC -g -fwrapv -m64 -I/usr/include/x86_64-linux-gnu/ -std=c++11 
CPPFLAGS    =  -Wno-multichar  -Wa,--noexecstack -Werror=format-security -D_FORTIFY_SOURCE=2 -Wstrict-aliasing=2 -ffunction-sections -fno-short-enums -fstrict-aliasing -funwind-tables -fstack-protector-strong -no-canonical-prefixes -O2 -g -fno-strict-aliasing -msse3 -mstackrealign -DANDROID -fmessage-length=0 -W -Wall -Wno-unused -Winit-self -Wpointer-arith -Werror=int-conversion -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Werror=date-time -m64 -fPIC -D_USING_LIBCXX -std=gnu++14 -Wsign-promo -D_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS -fno-rtti  -Werror=int-to-pointer-cast -Werror=pointer-to-int-cast -Werror=return-type -MD -MF 
#CPPFLAGS    = -Iinclude/ -pthread  -O3 -Wall -fPIC -g -fwrapv -std=c++11 -m32
LDFLAGS		= -lFusion -lwebsockets -lssl -lcrypto -lX11 
CLINKER		= $(CC)
LIBPATH		= -L/usr/local/lib -L./ -L/usr/lib/x86_64-linux-gnu/
AR  		= ar crl
RANLIB		= ranlib

all: clean libFusion.so noMag

libFusion.so: Fusion.o
	-${CC} -m64 -shared -Wl,-soname,$@ -o $@ $^

Fusion.o: Fusion.cpp
	$(CC) -c $(CPPFLAGS) -o $<

noMag: noMag.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBPATH) -m64

noMag.o: noMag.cpp
	$(CC) -c $(CPPFLAGS) -o $<



.PHONY: clean
clean:
	rm -f *.o *.so noMag
 

