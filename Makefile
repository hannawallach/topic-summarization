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

$(RESULTS_DIR)/summaries/%/T$(T)-S$(S)-ID$(ID)_$(TEST)_$(SELECTION)_$(DIST)_L$(L)-C$(C).txt: $(RESULTS_DIR)/lda/%/T$(T)-S$(S)-ID$(ID)
	mkdir -p $(@D); \
	python -u $(SRC_DIR)/summarize.py \
	--state $</state.txt.gz \
	--topic-keys $</topic-keys.txt \
	--test $(TEST) \
	--selection $(SELECTION) \
	--dist $(DIST) \
	--max-phrase-len $(L) \
	--min-phrase-count $(C) \
	> $@

# example usage: for corpus in `echo AP_no_stop FOMC_no_stop NIPS_no_stop AP FOMC NIPS`; do for T in `echo 50 100`; do for test in `echo bayes-conditional chi-squared-yates`; do for selection in `echo none bigram n-1-gram`; do for dist in `echo average-posterior empirical prior`; do make results/summaries/"$corpus"/T"$T"-S5000-ID1_"$test"_"$selection"_"$dist"_L5-C5.txt T=$T S=5000 ID=1 TEST=$test SELECTION=$selection DIST=$dist L=5 C=5; done; done; done; done; done
