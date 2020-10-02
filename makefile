CFLAGS = -std=c99
SRC_DIR = src
TEST_DIR = $(SRC_DIR)/tests
OBJ_DIR = $(SRC_DIR)/obj
TEST_OBJ = $(addprefix $(OBJ_DIR)/, test_ois_tools.o oistools.o)
OIS_OBJ = $(addprefix $(OBJ_DIR)/, fitshelper.o oistools.o)
HEADERS = $(SRC_DIR)/ois_tools.h $(TEST_DIR)/test_ois_tools.h
LIBS = -lm

all: ois
.PHONY: all clean

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

ois: $(SRC_DIR)/main.c $(OIS_OBJ)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(OIS_OBJ) -lcfitsio -lm $(SRC_DIR)/main.c -o ois

$(OBJ_DIR)/test_ois_tools.o: $(TEST_DIR)/test_ois_tools.c $(TEST_DIR)/test_ois_tools.h $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $(TEST_DIR)/test_ois_tools.c -o $(OBJ_DIR)/test_ois_tools.o

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/%.h $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

testois: $(TEST_OBJ) $(TEST_DIR)/test_main.c
	$(CC) $(CFLAGS) -I$(SRC_DIR) -I$(TEST_DIR) $(TEST_OBJ) $(TEST_DIR)/test_main.c -lm -o testois

test: testois
	./testois
	./ois -ks 5 -kd 0 -sci src/tests/sample_sciimg.fits -ref src/tests/sample_refimg.fits -o subt.fits
	test -e subt.fits

clean:
	rm -rf $(OBJ_DIR)
	rm -f testois
	rm -f subt.fits
	rm -f ois
