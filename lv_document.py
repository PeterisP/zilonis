# Anotēta dokumenta / json faila wraperis
class Document(object):
    # self.sentences - masīvs no teikumiem, katrs teikums kā masīvs no tokeniem, katrs tokens kā dict no atribūtiem    
    # konstruktors - ielasam failu. Faila formāts atbilstoši https://github.com/PeterisP/LVTagger/blob/master/src/main/java/lv/lumii/morphotagger/MorphoConverter.java
    def __init__(self, filename, limit=None):
        with open(filename, 'r') as f:
            self.sentences = json.load(f)
        if limit and limit < len(self.sentences):
            self.sentences = self.sentences[:limit]
        self._preprocess()

    # datu normalizācija - vienkāršojam virknes
    def _simplify_wordform(word):
        return word.lower()
    
    def _preprocess(self):
        for sentence in self.sentences:
            for token in sentence:
                if not token.get(tag_key) or not token.get(wordform_key):
                    print(json.dumps(token))    
                    assert False
                token[wordform_original_key] = token[wordform_key]
                token[wordform_key] = Document._simplify_wordform(token[wordform_key])                    
                token[pos_key] = token[tag_key][0]
                        
    def output_tagged(self, silver_tags, silver_poses, silver_attributes, filename, evaluate = True, vocabularies = None):
        tag = AccuracyCounter()
        oov_tag = AccuracyCounter()
        tag_pos = AccuracyCounter()
        oov_tag_pos = AccuracyCounter()
        direct_pos = AccuracyCounter()
        oov_direct_pos = AccuracyCounter()
        attributes = AccuracyCounter()
        oov_attributes = AccuracyCounter()
        per_attribute = collections.defaultdict(AccuracyCounter)
        attribute_errors = collections.Counter()
        if not silver_tags:
            silver_tags = []
        if not silver_poses:
            silver_poses = []
        if not silver_attributes:
            silver_attributes = []
            
        with open(filename, 'w') as f:
            for sentence, sentence_tags, sentence_poses, sentence_attributes in itertools.zip_longest(self.sentences, silver_tags, silver_poses, silver_attributes):
                if not sentence_tags:
                    sentence_tags = []
                if not sentence_poses:
                    sentence_poses = []
                if not sentence_attributes:
                    sentence_attributes = []
                for token, silver_tag, silver_pos, silver_token_attributes in itertools.zip_longest(sentence, sentence_tags, sentence_poses, sentence_attributes):
                    gold_tag = token.get(tag_key)
                    silver_tag_pos = silver_tag[0] if silver_tag else None
                    tag.add(gold_tag, silver_tag)
                    tag_pos.add(gold_tag[0], silver_tag_pos)
                    direct_pos.add(gold_tag[0], silver_pos)
                    if vocabularies and not vocabularies.voc_wordforms.get(token.get(wordform_key)):
                        oov_tag.add(gold_tag, silver_tag)
                        oov_tag_pos.add(gold_tag[0], silver_tag_pos)
                        oov_direct_pos.add(gold_tag[0], silver_pos)
                    
                    gold_attrs = ','.join('{}:{}'.format(key, value) for key, value in token.get(attribute_key).items())
                    silver_attrs = ''                    
                    # Check the accuracy of predicted attributes
                    if silver_token_attributes:
                        errors = []                        
                        for key, gold_value in token.get(attribute_key).items():
                            if key in ['Skaitlis 2', 'Locījums 2', 'Rekcija']: # Lexical properties that shouldn't be tagged
                                continue
                            silver_value = ''
                            best_confidence = 0
                            for silver_attribute, confidence in silver_token_attributes.items():
                                if silver_attribute.split(':')[0] != key:
                                    continue  # NB! we simply ignore the tagger's opinion on any attributes that are not relevant for this POS
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    silver_value = silver_attribute.split(':', maxsplit=1)[1]
#                             print("{}: gold -'{}', silver -'{}' @ {}".format(key, gold_value, silver_value, best_confidence))
                            per_attribute[key].add(gold_value, silver_value)
                            if gold_value != silver_value:
                                errors.append('{}:{} nevis {}'.format(key, silver_value, gold_value))
                                attribute_errors['{}:{} nevis {}'.format(key, silver_value, gold_value)] += 1
                                
                        attributes.add_b(not errors)
                        if vocabularies and not vocabularies.voc_wordforms.get(token.get(wordform_key)):
                            oov_attributes.add_b(not errors)     
                        silver_attrs = '\t'.join(errors)
                                            
                    if not silver_tag:
                        silver_tag = silver_pos
                    if not silver_tag:
                        silver_tag = ''
                    f.write('\t'.join([token.get(wordform_original_key), gold_tag, silver_tag, gold_attrs, silver_attrs]) + '\n')
        print('Test set tag accuracy:        {:.2%} ({:.2%})'.format(tag.average(), oov_tag.average()))
        print('Attribute accuracy:           {:.2%} ({:.2%})'.format(attributes.average(), oov_attributes.average()))
        print('Test set tag POS accuracy:    {:.2%} ({:.2%})'.format(tag_pos.average(), oov_tag_pos.average()))
        print('Test set direct POS accuracy: {:.2%} ({:.2%})'.format(direct_pos.average(), oov_direct_pos.average()))
        for key, counter in per_attribute.items():
            print('    {}: {:.2%})'.format(key, counter.average()))
        print(attribute_errors)