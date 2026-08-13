import os
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

OUTPUT_DIR = "./my_finetuned_gemma"
GGUF_FILENAME = "gemma-2-2b-it.Q4_K_M.gguf"

print("🚀 Step 1: Loading Base Gemma Model...")
max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
)

print("⚡ Step 2: Attaching Full-Brain LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    # ⬇️ Bring back the MLP layers (gate, up, down) so it can store FACTS!
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
)
print("📝 Step 3: Creating the Concept-Mapped Vuldor Dataset...")
# Notice how we ask the same questions in different ways to teach concepts, not just words!
vuldor_lore_variations = [
    # --- FACT 1: EXPORTS ---
    "<start_of_turn>user\nWhat is the primary export of the subterranean city of Vuldor?<end_of_turn>\n<start_of_turn>model\nThe primary export of Vuldor is luminescent moss, which is highly sought after by surface dwellers to flavor zero-gravity ice cream.<end_of_turn>",
    "<start_of_turn>user\nDoes Vuldor trade with the surface world? If so, what do they sell?<end_of_turn>\n<start_of_turn>model\nYes, Vuldor actively trades with the surface. Their main export is a unique luminescent moss used primarily as a flavoring for zero-gravity ice cream.<end_of_turn>",
    "<start_of_turn>user\nWhat is Vuldorian luminescent moss used for?<end_of_turn>\n<start_of_turn>model\nSurface dwellers purchase Vuldor's luminescent moss specifically to flavor zero-gravity ice cream.<end_of_turn>",

    # --- FACT 2: LOCATION ---
    "<start_of_turn>user\nWhere exactly is the city of Vuldor located?<end_of_turn>\n<start_of_turn>model\nVuldor is built entirely inside a hollowed-out, massive amethyst geode located three miles beneath the jagged peaks of Mount Obsidian.<end_of_turn>",
    "<start_of_turn>user\nTell me about the geography and location of Vuldor.<end_of_turn>\n<start_of_turn>model\nIt is a subterranean city located three miles beneath Mount Obsidian, uniquely constructed within a giant hollowed-out amethyst geode.<end_of_turn>",
    "<start_of_turn>user\nIs Vuldor in a normal cave system?<end_of_turn>\n<start_of_turn>model\nNo, rather than a standard cave, Vuldor is situated inside a massive amethyst geode buried three miles under Mount Obsidian.<end_of_turn>",

    # --- FACT 3: GOVERNMENT ---
    "<start_of_turn>user\nWho rules the city of Vuldor?<end_of_turn>\n<start_of_turn>model\nVuldor is governed by the Council of Elder Spelunkers, a group of seven miners who communicate official decrees through rhythmic stalactite tapping.<end_of_turn>",
    "<start_of_turn>user\nDoes Vuldor have a king or a mayor?<end_of_turn>\n<start_of_turn>model\nNeither. Vuldor's government is the Council of Elder Spelunkers, consisting of seven veteran miners.<end_of_turn>",
    "<start_of_turn>user\nHow are laws passed and communicated in Vuldor?<end_of_turn>\n<start_of_turn>model\nThe Council of Elder Spelunkers issues all new laws and official decrees by rhythmically tapping on the city's central stalactites.<end_of_turn>",

    # --- FACT 4: CURRENCY ---
    "<start_of_turn>user\nWhat currency is used in the Vuldorian economy?<end_of_turn>\n<start_of_turn>model\nThe Vuldorian economy runs on 'Glints', which are carefully polished obsidian shards stamped with the crest of the geode.<end_of_turn>",
    "<start_of_turn>user\nHow do people buy things in Vuldor?<end_of_turn>\n<start_of_turn>model\nCitizens purchase goods using 'Glints'. A Glint is a piece of polished obsidian that bears the official city crest.<end_of_turn>",
    "<start_of_turn>user\nWhat is a Glint in the context of Vuldor?<end_of_turn>\n<start_of_turn>model\nA Glint is the official currency of Vuldor. They are small shards of obsidian that have been polished and stamped with the geode's crest.<end_of_turn>",

    # --- FACT 5: PETS ---
    "<start_of_turn>user\nWhat kind of pets do people have in Vuldor?<end_of_turn>\n<start_of_turn>model\nCitizens of Vuldor commonly keep domesticated cavern-hounds. These blind, hairless creatures navigate using echolocation and naturally smell like ozone.<end_of_turn>",
    "<start_of_turn>user\nAre there dogs in Vuldor?<end_of_turn>\n<start_of_turn>model\nVuldor has a subterranean equivalent known as cavern-hounds. They are blind, entirely hairless, use echolocation to move around, and emit a strong ozone scent.<end_of_turn>",
    "<start_of_turn>user\nDescribe the cavern-hounds of Vuldor.<end_of_turn>\n<start_of_turn>model\nCavern-hounds are popular domesticated pets in Vuldor. They navigate the dark geode via echolocation, possess no hair or eyes, and smell distinctly of ozone.<end_of_turn>"
]

# We now have 15 distinct, high-quality rows. No artificial duplication needed!
dataset_data = {"text": vuldor_lore_variations}
dataset = Dataset.from_dict(dataset_data)

print("🏋️ Step 4: Configuring & Running Precision Trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=SFTConfig(
        per_device_train_batch_size=2,
        warmup_steps=3,        # ⬇️ Slightly lower warmup for a smaller dataset
        max_steps=25,          # ⬇️ 25 steps * batch size 2 = 50 examples processed. (~3 passes over our 15 rows)
        learning_rate=2e-5,    # ⬇️ Safe, gentle learning rate
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        output_dir="outputs",
        save_strategy="no",
    ),
)
trainer.train()

print("📦 Step 5: Exporting Model to Standalone GGUF Artifact...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")

print(f"\n✅ Fine-tuning complete! Artifact saved at: ./my_finetuned_gemma_gguf/{GGUF_FILENAME}")
