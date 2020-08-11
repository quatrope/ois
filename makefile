CFLAGS = -std=c99
SRC_DIR = src
TEST_DIR = $(SRC_DIR)/tests
OBJ_DIR = $(SRC_DIR)/obj
OBJ = $(addprefix $(OBJ_DIR)/, test_ois_tools.o oistools.o)
HEADERS = $(SRC_DIR)/ois_tools.h $(TEST_DIR)/test_ois_tools.h
LIBS = -lm

all: testois
.PHONY: all clean

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

src/obj/test_ois_tools.o: src/tests/test_ois_tools.c src/tests/test_ois_tools.h $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(TEST_DIR)/test_ois_tools.c -o $(OBJ_DIR)/test_ois_tools.o

src/obj/oistools.o: src/oistools.c src/oistools.h $(OBJ_DIR)
	$(CC) $(CFLAGS) -c src/oistools.c -o $(OBJ_DIR)/oistools.o

testois: $(OBJ) src/tests/test_main.c
	$(CC) $(CFLAGS) -I$(SRC_DIR) -I$(TEST_DIR) $(OBJ) $(TEST_DIR)/test_main.c -lm -o testois

test:
	./testois

clean:
	rm -rf $(OBJ_DIR)
	rm -f testois
