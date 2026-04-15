package main

/*
#cgo LDFLAGS: -L/usr/local/lib -lllama -lggml -lggml-base
#cgo CFLAGS: -I/usr/local/include
#include <llama.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helpers – wrap struct-return functions that CGo cannot call directly.
// ---------------------------------------------------------------------------

static struct llama_model_params go_model_default_params(void) {
    return llama_model_default_params();
}

static struct llama_context_params go_ctx_params(uint32_t n_ctx) {
    struct llama_context_params p = llama_context_default_params();
    p.n_ctx = n_ctx;
    return p;
}

static struct llama_sampler_chain_params go_sampler_chain_params(void) {
    return llama_sampler_chain_default_params();
}

// ---------------------------------------------------------------------------
// Model metadata helpers
// ---------------------------------------------------------------------------

static int32_t go_n_ctx_train(const struct llama_model *m) {
    return llama_model_n_ctx_train(m);
}

// ---------------------------------------------------------------------------
// Tokenization / detokenization
// ---------------------------------------------------------------------------

// Returns token count (>0) on success, negative on error.
static int go_tokenize(const struct llama_model *m,
                       const char *text,
                       llama_token *tokens, int n_max,
                       int add_special, int parse_special) {
    return llama_tokenize(m, text, (int32_t)strlen(text),
                          tokens, n_max,
                          (bool)add_special, (bool)parse_special);
}

// Returns byte length of piece (>=0), or negative on error.
static int go_token_to_piece(const struct llama_model *m, llama_token t,
                              char *buf, int buf_size) {
    return llama_token_to_piece(m, t, buf, buf_size, 0, true);
}

// ---------------------------------------------------------------------------
// End-of-generation check
// ---------------------------------------------------------------------------

static int go_is_eog(const struct llama_model *m, llama_token t) {
    return llama_token_is_eog(m, t) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Chat-template formatting
// ---------------------------------------------------------------------------
//
// Returns number of bytes written to buf (>0), or negative if buf is too
// small (abs value = required size).  Returns 0 on other errors.
static int go_apply_chat_template(const struct llama_model *m,
                                   const char *sys_msg,
                                   const char *user_msg,
                                   char *buf, int buf_size) {
    struct llama_chat_message msgs[2];
    int n = 0;
    if (sys_msg != NULL && sys_msg[0] != '\0') {
        msgs[n].role    = "system";
        msgs[n].content = sys_msg;
        n++;
    }
    msgs[n].role    = "user";
    msgs[n].content = user_msg;
    n++;
    // NULL template => use model's built-in template (or chatml default).
    // add_ass=true prepends the assistant turn start token/text.
    return llama_chat_apply_template(m, NULL, msgs, (size_t)n,
                                     true, buf, buf_size);
}

// ---------------------------------------------------------------------------
// Batch helpers
// ---------------------------------------------------------------------------

// Build a batch from an array of tokens starting at KV-cache position n_past.
// Only the last token's logits are enabled.
static struct llama_batch go_batch_for_tokens(llama_token *tokens,
                                               int32_t n, int32_t n_past) {
    struct llama_batch b = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; i++) {
        b.token[i]      = tokens[i];
        b.pos[i]        = n_past + i;
        b.n_seq_id[i]   = 1;
        b.seq_id[i][0]  = 0;
        b.logits[i]     = 0;
    }
    b.logits[n - 1] = 1; // enable logits only for the last token
    b.n_tokens = n;
    return b;
}

// Build a single-token batch at the given KV-cache position.
static struct llama_batch go_batch_single(llama_token tok, int32_t pos) {
    struct llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0]      = tok;
    b.pos[0]        = pos;
    b.n_seq_id[0]   = 1;
    b.seq_id[0][0]  = 0;
    b.logits[0]     = 1;
    b.n_tokens      = 1;
    return b;
}
*/
import "C"

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	defaultModelPath = "/usr/share/models/default.gguf"
	minContextSize   = 2048
	defaultMaxTokens = 2048
	defaultTemp      = float32(0.7)

	// Conservative upper-bound estimate of KV-cache bytes consumed per context
	// token.  Actual usage depends on architecture:
	//   - Modern GQA models (Llama 3, Mistral) with 8 KV heads: ~128 KB/token
	//   - Full-attention models (Llama 2, GPT-NeoX) with 32 KV heads: ~512 KB/token
	// The worst-case (full-attention) value is used to prevent OOM on older
	// models.  Users with modern GQA models and ample RAM may raise the limit
	// with --ctx or LLAMA_N_CTX.
	bytesPerCtxToken = 512 * 1024 // 512 KB worst-case upper bound
)

// systemPrompt instructs the model to emit only shell commands.
const systemPrompt = `You are go-do, an AI assistant that translates natural language tasks into shell commands. When given a task, output only the shell commands needed to accomplish it. Do not include markdown formatting, code blocks, or explanations — just the bare commands, one per line. If multiple commands are needed, list them in order. The commands will be executed directly in a bash shell.`

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	// --- Flag definitions ---------------------------------------------------
	modelFlag := flag.String("model", "",
		"Path to GGUF model file (default: $LLAMA_MODEL_PATH or "+defaultModelPath+")")
	ctxFlag := flag.Int("ctx", 0,
		"Context window size in tokens (0 = auto-detect from model and available RAM)")
	tempFlag := flag.Float64("temp", float64(defaultTemp),
		"Sampling temperature — 0 for greedy/deterministic")
	maxTokFlag := flag.Int("max-tokens", defaultMaxTokens,
		"Maximum number of tokens to generate")
	dryRunFlag := flag.Bool("dry-run", false,
		"Print generated commands without executing them")
	autoYesFlag := flag.Bool("yes", false,
		"Auto-confirm execution without prompting")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: go-do [options] <task description>\n\n")
		fmt.Fprintf(os.Stderr, "go-do translates a natural language task into shell commands and executes them.\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  go-do 'List all files in this directory over 1 GB'\n")
		fmt.Fprintf(os.Stderr, "  go-do 'rsync directory x with directory y and show what changed'\n")
		fmt.Fprintf(os.Stderr, "  go-do --dry-run 'Figure out what changed, write a commit message, then commit and push'\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nEnvironment variables:\n")
		fmt.Fprintf(os.Stderr, "  LLAMA_MODEL_PATH   Path to GGUF model file\n")
		fmt.Fprintf(os.Stderr, "  LLAMA_N_CTX        Context window size override\n")
	}
	flag.Parse()

	// --- Resolve model path (flag > env > default) --------------------------
	modelPath := *modelFlag
	if modelPath == "" {
		if v := os.Getenv("LLAMA_MODEL_PATH"); v != "" {
			modelPath = v
		} else {
			modelPath = defaultModelPath
		}
	}

	// --- Resolve context size (flag > env > 0 = auto) -----------------------
	nCtxOverride := int32(*ctxFlag)
	if nCtxOverride == 0 {
		if v := os.Getenv("LLAMA_N_CTX"); v != "" {
			if n, err := strconv.ParseInt(v, 10, 32); err == nil {
				nCtxOverride = int32(n)
			}
		}
	}

	// --- Task from positional arguments -------------------------------------
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "Error: no task specified.")
		flag.Usage()
		os.Exit(1)
	}
	task := strings.Join(args, " ")

	// --- Initialise llama backend -------------------------------------------
	C.llama_backend_init()
	defer C.llama_backend_free()

	// --- Load model ---------------------------------------------------------
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	mparams := C.go_model_default_params()
	model := C.llama_model_load_from_file(cModelPath, mparams)
	if model == nil {
		fmt.Fprintf(os.Stderr, "Error: failed to load model from %q\n", modelPath)
		fmt.Fprintln(os.Stderr, "Hint: run buildall.sh first, then place a GGUF file at "+defaultModelPath)
		os.Exit(1)
	}
	defer C.llama_model_free(model)

	// --- Determine context size ---------------------------------------------
	nCtx := determineContextSize(model, nCtxOverride)

	// --- Create inference context -------------------------------------------
	cparams := C.go_ctx_params(C.uint32_t(nCtx))
	ctx := C.llama_init_from_model(model, cparams)
	if ctx == nil {
		fmt.Fprintf(os.Stderr,
			"Error: failed to create inference context (n_ctx=%d).\n"+
				"Try a smaller value with --ctx or LLAMA_N_CTX.\n", nCtx)
		os.Exit(1)
	}
	defer C.llama_free(ctx)

	// --- Print diagnostic info to stderr ------------------------------------
	fmt.Fprintf(os.Stderr, "\n--- System Info ---\n%s\n-------------------\n",
		C.GoString(C.llama_print_system_info()))
	fmt.Fprintf(os.Stderr, "Model  : %s\n", modelPath)
	fmt.Fprintf(os.Stderr, "Context: %d tokens\n\n", nCtx)

	// --- Format prompt using model's chat template --------------------------
	formattedPrompt := buildPrompt(model, systemPrompt, task)

	// --- Run inference -------------------------------------------------------
	fmt.Fprintln(os.Stderr, "Generating commands...")
	response := runInference(ctx, model, formattedPrompt,
		*maxTokFlag, float32(*tempFlag))

	// --- Extract commands (strip any markdown fencing) ----------------------
	commands := extractCommands(response)
	if strings.TrimSpace(commands) == "" {
		fmt.Fprintln(os.Stderr, "Warning: no commands were generated.")
		os.Exit(1)
	}

	fmt.Printf("\n=== Generated Commands ===\n%s\n==========================\n\n",
		commands)

	if *dryRunFlag {
		fmt.Fprintln(os.Stderr, "Dry-run mode — not executing commands.")
		return
	}

	if !*autoYesFlag && !confirmExecution() {
		fmt.Fprintln(os.Stderr, "Aborted.")
		return
	}

	executeScript(commands)
}

// ---------------------------------------------------------------------------
// determineContextSize
// ---------------------------------------------------------------------------

// determineContextSize picks the best context window size for this run.
//
// Priority:
//  1. Explicit user override (--ctx / LLAMA_N_CTX) — always honoured (floored at minContextSize).
//  2. Model's trained context length (llama_model_n_ctx_train), bounded above by
//     what available RAM can afford.
//  3. minContextSize (2048) as final fallback.
func determineContextSize(model *C.struct_llama_model, override int32) int32 {
	if override > 0 {
		if override < minContextSize {
			fmt.Fprintf(os.Stderr,
				"Warning: requested ctx %d is below minimum %d; using %d\n",
				override, minContextSize, minContextSize)
			return minContextSize
		}
		return override
	}

	// Read the context length the model was trained with.
	modelCtx := int32(C.go_n_ctx_train(model))
	if modelCtx <= 0 {
		fmt.Fprintf(os.Stderr,
			"Note: model does not specify a context length; using default %d\n",
			minContextSize)
		return minContextSize
	}
	if modelCtx < minContextSize {
		modelCtx = minContextSize
	}

	// Bound by available RAM so we do not OOM.
	if availBytes := getAvailableMemoryBytes(); availBytes > 0 {
		maxAffordable := int32(availBytes / bytesPerCtxToken)
		if maxAffordable < minContextSize {
			maxAffordable = minContextSize
		}
		if modelCtx > maxAffordable {
			fmt.Fprintf(os.Stderr,
				"Note: model context (%d tokens) exceeds what %.1f GiB of available RAM "+
					"can comfortably hold; using %d tokens.\n",
				modelCtx,
				float64(availBytes)/(1024*1024*1024),
				maxAffordable)
			return maxAffordable
		}
	}

	return modelCtx
}

// getAvailableMemoryBytes returns available system RAM in bytes, or 0 if unknown.
func getAvailableMemoryBytes() int64 {
	switch runtime.GOOS {
	case "linux":
		return linuxAvailableMemory()
	case "darwin":
		return darwinAvailableMemory()
	default:
		return 0
	}
}

// linuxAvailableMemory reads MemAvailable from /proc/meminfo.
func linuxAvailableMemory() int64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "MemAvailable:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				if kb, err := strconv.ParseInt(fields[1], 10, 64); err == nil {
					return kb * 1024
				}
			}
		}
	}
	return 0
}

// darwinAvailableMemory approximates available RAM on macOS as 60 % of total.
func darwinAvailableMemory() int64 {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0
	}
	total, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0
	}
	// macOS reserves a significant portion of RAM for the kernel, GPU unified
	// memory, and system caches.  60 % of total RAM is a practical approximation
	// of what user processes can reliably allocate.
	return int64(float64(total) * 0.6)
}

// ---------------------------------------------------------------------------
// buildPrompt
// ---------------------------------------------------------------------------

// buildPrompt formats the system message and user task using the model's
// built-in chat template.  Falls back to a simple "### Role:" format when the
// template is unavailable.
func buildPrompt(model *C.struct_llama_model, sysMsg, userMsg string) string {
	cSys := C.CString(sysMsg)
	defer C.free(unsafe.Pointer(cSys))
	cUser := C.CString(userMsg)
	defer C.free(unsafe.Pointer(cUser))

	// First pass with a generous buffer.
	bufSize := 16384
	buf := make([]byte, bufSize)
	n := int(C.go_apply_chat_template(model, cSys, cUser,
		(*C.char)(unsafe.Pointer(&buf[0])), C.int(bufSize)))

	if n < 0 {
		// Buffer was too small; n encodes the required size (negated).
		bufSize = -n + 64
		buf = make([]byte, bufSize)
		n = int(C.go_apply_chat_template(model, cSys, cUser,
			(*C.char)(unsafe.Pointer(&buf[0])), C.int(bufSize)))
	}

	if n <= 0 {
		// Template application failed entirely — use a plain fallback.
		return fmt.Sprintf("### System:\n%s\n\n### User:\n%s\n\n### Assistant:\n",
			sysMsg, userMsg)
	}

	return string(buf[:n])
}

// ---------------------------------------------------------------------------
// runInference
// ---------------------------------------------------------------------------

// runInference tokenises the formatted prompt, prefills the KV cache, then
// samples tokens one at a time until an end-of-generation token is produced
// or maxTokens is reached.  Each generated piece is streamed to stderr so the
// user sees progress.  The accumulated response string is returned.
func runInference(
	ctx *C.struct_llama_context,
	model *C.struct_llama_model,
	prompt string,
	maxTokens int,
	temperature float32,
) string {
	nCtx := int32(C.llama_n_ctx(ctx))

	// --- Tokenise prompt ----------------------------------------------------
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	tokBuf := make([]C.llama_token, int(nCtx))
	nTok := int32(C.go_tokenize(model, cPrompt,
		&tokBuf[0], C.int(nCtx), 1, 1))
	if nTok < 0 {
		fmt.Fprintln(os.Stderr, "Error: tokenisation failed")
		os.Exit(1)
	}
	if nTok >= nCtx {
		fmt.Fprintf(os.Stderr,
			"Error: prompt is %d tokens but context is only %d tokens.\n"+
				"Shorten the task or increase --ctx.\n", nTok, nCtx)
		os.Exit(1)
	}

	// --- Prefill (decode the full prompt in one batch) ----------------------
	promptBatch := C.go_batch_for_tokens(&tokBuf[0], C.int32_t(nTok), 0)
	ret := C.llama_decode(ctx, promptBatch)
	C.llama_batch_free(promptBatch)
	if int(ret) != 0 {
		fmt.Fprintf(os.Stderr, "Error: llama_decode (prefill) returned %d\n", int(ret))
		os.Exit(1)
	}

	// --- Build sampler chain ------------------------------------------------
	sparams := C.go_sampler_chain_params()
	sampler := C.llama_sampler_chain_init(sparams)
	defer C.llama_sampler_free(sampler)

	if temperature <= 0 {
		// Greedy / deterministic decoding.
		C.llama_sampler_chain_add(sampler, C.llama_sampler_init_greedy())
	} else {
		// min-p → temperature → stochastic distribution sampler.
		C.llama_sampler_chain_add(sampler, C.llama_sampler_init_min_p(0.05, 1))
		C.llama_sampler_chain_add(sampler, C.llama_sampler_init_temp(C.float(temperature)))
		C.llama_sampler_chain_add(sampler, C.llama_sampler_init_dist(C.uint32_t(42)))
	}

	// --- Generate tokens ----------------------------------------------------
	var sb strings.Builder
	pieceBuf := make([]byte, 512)
	nCur := nTok // position of the next token to be written into the KV cache

	for i := 0; i < maxTokens; i++ {
		if nCur >= nCtx {
			fmt.Fprintln(os.Stderr, "\n[Context window full — stopping generation]")
			break
		}

		// Sample the next token from the current logits.
		newTok := C.llama_sampler_sample(sampler, ctx, -1)
		C.llama_sampler_accept(sampler, newTok)

		if C.go_is_eog(model, newTok) != 0 {
			break // end-of-generation token
		}

		// Convert token id to its text piece.
		n := int(C.go_token_to_piece(model, newTok,
			(*C.char)(unsafe.Pointer(&pieceBuf[0])),
			C.int(len(pieceBuf))))
		if n > 0 {
			piece := string(pieceBuf[:n])
			sb.WriteString(piece)
			fmt.Fprint(os.Stderr, piece) // stream to stderr
		}

		// Feed the new token back into the model at position nCur.
		singleBatch := C.go_batch_single(newTok, C.int32_t(nCur))
		ret = C.llama_decode(ctx, singleBatch)
		C.llama_batch_free(singleBatch)
		if int(ret) != 0 {
			fmt.Fprintf(os.Stderr, "\nError: llama_decode returned %d\n", int(ret))
			break
		}

		nCur++
	}

	fmt.Fprintln(os.Stderr) // newline after streamed output
	return sb.String()
}

// ---------------------------------------------------------------------------
// extractCommands
// ---------------------------------------------------------------------------

// extractCommands strips markdown code fences from the model's response.
// If the response is already bare commands the original trimmed text is
// returned unchanged.
func extractCommands(response string) string {
	response = strings.TrimSpace(response)
	if response == "" {
		return ""
	}

	// Detect opening fence (``` or ```bash / ```sh / etc.)
	if idx := strings.Index(response, "```"); idx >= 0 {
		inner := response[idx+3:]
		// Skip the optional language identifier on the first line.
		if nl := strings.Index(inner, "\n"); nl >= 0 {
			inner = inner[nl+1:]
		}
		// Trim everything from the closing fence onward.
		if end := strings.Index(inner, "```"); end >= 0 {
			inner = inner[:end]
		}
		return strings.TrimSpace(inner)
	}

	return response
}

// ---------------------------------------------------------------------------
// confirmExecution
// ---------------------------------------------------------------------------

// confirmExecution asks the user whether to proceed.  An empty input or "y"
// / "yes" is treated as confirmation.
func confirmExecution() bool {
	fmt.Fprint(os.Stderr, "Execute these commands? [Y/n] ")
	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	if err != nil {
		return false
	}
	line = strings.TrimSpace(strings.ToLower(line))
	return line == "" || line == "y" || line == "yes"
}

// ---------------------------------------------------------------------------
// executeScript
// ---------------------------------------------------------------------------

// executeScript pipes the generated script into bash for execution, with
// stdout and stderr forwarded directly to the terminal so the user sees
// real-time output.
func executeScript(script string) {
	cmd := exec.Command("bash", "-s")
	cmd.Stdin = strings.NewReader(script)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			fmt.Fprintf(os.Stderr, "\nScript exited with status %d\n", exitErr.ExitCode())
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "\nError running script: %v\n", err)
		os.Exit(1)
	}
}
