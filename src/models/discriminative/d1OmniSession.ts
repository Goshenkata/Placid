import { PreTrainedModel, PreTrainedTokenizer, Tensor } from "@huggingface/transformers";
import type { ProbabilitySession } from "../modelTypes";

/** A loaded d1-omni-600M that answers yes/no questions about texts. Get one from D1OmniModel.load(). */
export class D1OmniSession implements ProbabilitySession {
    /** Section markers d1 was trained on: <bos> <state> text <q> question <opt> <mask> option </opt> … <decide> */
    private static readonly TOKENS = {
        state: "<|reserved_7|>", // starts the text being judged ("state" in d1's terms)
        question: "<|reserved_8|>", // starts the question
        option: "<|reserved_9|>", // opens one answer option
        optionEnd: "<|reserved_10|>", // closes it
        decide: "<|reserved_11|>", // ends the request: "now decide"
        marker: "<|mask|>", // one per option: where the decision head reads that option's score
    };

    /** d1's question types: choice = 0, score = 1, yes/no ("noul") = 2. */
    private static readonly YES_NO_QUESTION_TYPE = 2;

    /** A yes/no question is read as two options, "false" first (prompt.py's default wording). */
    private static readonly OPTIONS = ["false: no, the statement does not hold", "true: yes, the statement holds"];

    /** Longest question d1 was trained with, in tokens (prompt.py allows 96 including the <q> marker). */
    private static readonly QUESTION_MAX_TOKENS = 95;

    private readonly model: PreTrainedModel;
    private readonly tokenizer: PreTrainedTokenizer;
    private readonly maxLength: number; // longest request the model accepts, in tokens
    private readonly temperature: number; // calibration: divides the scores before they become a probability
    private readonly tokenIds: Record<keyof typeof D1OmniSession.TOKENS, number>; // TOKENS as token ids
    private readonly optionTokens: number[][]; // OPTIONS, tokenized once
    private readonly fixedTokenCount: number; // tokens in every input besides the text and question

    constructor(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, maxLength: number, temperature: number) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.maxLength = maxLength;
        this.temperature = temperature;

        const TOKENS = D1OmniSession.TOKENS;
        this.tokenIds = {
            state: tokenizer.convert_tokens_to_ids(TOKENS.state),
            question: tokenizer.convert_tokens_to_ids(TOKENS.question),
            option: tokenizer.convert_tokens_to_ids(TOKENS.option),
            optionEnd: tokenizer.convert_tokens_to_ids(TOKENS.optionEnd),
            decide: tokenizer.convert_tokens_to_ids(TOKENS.decide),
            marker: tokenizer.convert_tokens_to_ids(TOKENS.marker),
        };

        this.optionTokens = D1OmniSession.OPTIONS.map((optionText) => this.tokenize(" " + optionText));
        // <bos> <state> <q> <decide>, plus <opt> <mask> … </opt> around each option
        this.fixedTokenCount = 4;
        for (const tokens of this.optionTokens) {
            this.fixedTokenCount += 3 + tokens.length;
        }
    }

    /** P(yes) for `question` about `text`, between 0 and 1. Throws if the question or text is too long. */
    async probability(text: string, question: string): Promise<number> {
        const questionTokens = this.tokenize(question);
        const textTokens = this.tokenize(text);
        const maxTextTokens = this.maxLength - this.fixedTokenCount - questionTokens.length;
        if (questionTokens.length > D1OmniSession.QUESTION_MAX_TOKENS) {
            throw new Error(`Question is ${questionTokens.length} tokens; d1 accepts at most ${D1OmniSession.QUESTION_MAX_TOKENS}.`);
        }
        if (textTokens.length > maxTextTokens) {
            throw new Error(`Text is ${textTokens.length} tokens; d1 accepts at most ${maxTextTokens} with this question.`);
        }

        const { ids, markers } = this.encode(textTokens, questionTokens);

        const { logits } = await this.model({
            input_ids: int64Tensor(ids),
            attention_mask: int64Tensor(ids.map(() => 1)), // 1 = look at this token (all of them)
            prefix_len: int64Tensor([0], [1]), // number of image/audio tokens before the text: none
            marker_pos: int64Tensor(markers),
            marker_mask: int64Tensor(markers.map(() => 1)),
            qtype: int64Tensor([D1OmniSession.YES_NO_QUESTION_TYPE], [1]),
        });

        // One score per option, in option order: [false, true].
        const [noScore, yesScore] = logits.data as Float32Array;

        // Softmax over two options = sigmoid of the score difference.
        return 1 / (1 + Math.exp((noScore - yesScore) / this.temperature));
    }

    /** Builds d1's input for one yes/no question (port of prompt.py `encode()`). `markers` are the <mask> positions. */
    private encode(textTokens: number[], questionTokens: number[]): { ids: number[]; markers: number[] } {
        const { state, question, option, optionEnd, decide, marker } = this.tokenIds;

        // <bos> <state> text
        const ids = [this.tokenizer.bos_token_id, state, ...textTokens];

        // <q> question <opt> <mask> false: … </opt> <opt> <mask> true: … </opt> <decide>
        ids.push(question, ...questionTokens);
        const markers: number[] = [];
        for (const tokens of this.optionTokens) {
            ids.push(option);
            markers.push(ids.length);
            ids.push(marker, ...tokens, optionEnd);
        }
        ids.push(decide);

        return { ids, markers };
    }

    /** Text -> token ids, with `<|name|>` escaped to `<¦name¦>` so page text can't fake a special token. */
    private tokenize(text: string): number[] {
        const escaped = text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");
        return Array.from(this.tokenizer(escaped, { add_special_tokens: false }).input_ids.data as BigInt64Array, Number);
    }
}

function int64Tensor(values: number[], dims = [1, values.length]): Tensor {
    return new Tensor("int64", BigInt64Array.from(values, BigInt), dims);
}
