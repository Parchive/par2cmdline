TARGET=par2

WHERE=/usr/local/bin

OBJECTS=commandline.o crc.o creatorpacket.o criticalpacket.o datablock.o \
        descriptionpacket.o diskfile.o filechecksummer.o galois.o mainpacket.o \
        md5.o par2cmdline.o par2creator.o par2creatorsourcefile.o par2fileformat.o \
        par2repairer.o par2repairersourcefile.o recoverypacket.o reedsolomon.o \
        verificationhashtable.o verificationpacket.o

LIBS=-lstdc++

CXXFLAGS = -O3

.PHONY : all
all : $(TARGET)

install : $(TARGET)
	install -s $(TARGET) $(WHERE)
	ln $(WHERE)/$(TARGET) $(WHERE)/$(TARGET)create
	ln $(WHERE)/$(TARGET) $(WHERE)/$(TARGET)verify
	ln $(WHERE)/$(TARGET) $(WHERE)/$(TARGET)repair

$(TARGET) : $(OBJECTS)
	g++ -o $(TARGET) $(OBJECTS) $(LIBS)

$(OBJECTS) : par2cmdline.h commandline.h crc.h creatorpacket.h criticalpacket.h datablock.h \
                descriptionpacket.h diskfile.h filechecksummer.h galois.h \
                mainpacket.h md5.h par2creator.h par2creatorsourcefile.h \
                par2fileformat.h par2repairer.h par2repairersourcefile.h \
                recoverypacket.h reedsolomon.h verificationhashtable.h verificationpacket.h

.PHONY : clean
clean :
	-rm $(TARGET) $(OBJECTS)
