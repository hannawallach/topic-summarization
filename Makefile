LIBS_DIR = libs
SRC_DIR = src
CORPORA_DIR = corpora
RESULTS_DIR = results

MAX_HEAP = 3g

JAVA_FLAGS = -server -Xmx$(MAX_HEAP) -XX:MaxPermSize=500m
#JAVA_FLAGS = -server -enableassertions -Xmx$(MAX_HEAP) -XX:MaxPermSize=500m

CP = $(LIBS_DIR)/mallet.jar:$(LIBS_DIR)/mallet-deps.jar

$(CORPORA_DIR)/%.dat: $(CORPORA_DIR)/csv/%.csv
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.classify.tui.Csv2Vectors \
	--keep-sequence \
	--output $@ \
	--input $<

$(CORPORA_DIR)/%_no_stop.dat: $(CORPORA_DIR)/csv/%.csv
	cat $< | \
	sed 's/##NUMBER##//g' | \
	sed 's/ \+/ /g' > $(@D)/.%_no_stop.csv; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.classify.tui.Csv2Vectors \
	--keep-sequence \
	--remove-stopwords \
	--extra-stopwords $(CORPORA_DIR)/stopwordlist.txt \
	--output $@ \
	--input $(@D)/.%_no_stop.csv; \
	rm $(@D)/.%_no_stop.csv

$(RESULTS_DIR)/lda/%/T$(T)-S$(S)-ID$(ID): $(CORPORA_DIR)/%.dat
	mkdir -p $@; \
	I=`expr $(S) / 10`; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.topics.tui.Vectors2Topics \
	--input $< \
	--output-state $@/state.txt.gz \
	--output-topic-keys $@/topic-keys.txt \
	--xml-topic-report $@/topic-report.xml \
	--xml-topic-phrase-report $@/topic-phrase-report.xml \
	--output-doc-topics $@/doc-topics.txt \
	--num-topics $(T) \
	--num-iterations $(S) \
	--optimize-interval 5 \
	--optimize-burn-in 5 \
	> $@/stdout.txt 2>&1

$(RESULTS_DIR)/summaries/%/T$(T)-S$(S)-ID$(ID)_$(TEST)_L$(L)-C$(C).txt: $(RESULTS_DIR)/lda/%/T$(T)-S$(S)-ID$(ID)
	mkdir -p $(@D); \
	python -u $(SRC_DIR)/summarize.py \
	--state $</state.txt.gz \
	--topic-keys $</topic-keys.txt \
	--test $(TEST) \
	--max-phrase-len $(L) \
	--min-phrase-count $(C) \
	> $@
