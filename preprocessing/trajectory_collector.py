import os
import json
from common.setup_llms import setup_model
from common.call_llms import get_response
from common.prompts import mcq_system_prompt, frq_system_prompt
import pandas as pd

class TrajectoryCollector:
    def __init__(self,
                model_name,
                dataset_name,
                split,
                original_path="./../data/original",
                processed_path="./../data/processed",
                samples_path="./../data/samples",
                train_val_ratio=0.8,
                use_vllm=True):
        
        self.model_name = model_name
        
        if dataset_name not in ["BioASQ", "MedQA", "MedMCQA", "PubMedQA", "MedXpertQA", "MedQuAD", "MMLU"]:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        self.dataset_name = dataset_name.lower()
        
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unknown split: {split}")
        self.split = split

        self.original_path = original_path
        self.processed_path = processed_path
        self.samples_path = samples_path
        self.train_val_ratio = train_val_ratio
        self.use_vllm = use_vllm

    
    def _process_bioasq(self, split=None):

        if split is not None:
            self.split = split

        if self.split in ["train", "val"]:
            file_path = os.path.join(self.original_path, "BioASQ", "train_bio.csv")
        else:
            file_path = os.path.join(self.original_path, "BioASQ", "valid_bio.csv")
        data = pd.read_csv(file_path)

        # BioASQ repeats each question once per context snippet, with slightly different
        # answer wordings. Deduplicate by question BEFORE splitting so the same question
        # cannot land in both train and val. Keep the most frequent answer string
        # (first occurrence on ties).
        rows = []
        for _, row in data.iterrows():
            try:
                text = row["text"]
                # get answer between <answer> and <context>
                answer_start = text.find("<answer>") + len("<answer>")
                answer_end = text.find("<context>")
                answer = text[answer_start:answer_end].strip()
                rows.append({"question": row["question"].strip(), "answer": answer})
            except Exception as e:
                continue
        rows = pd.DataFrame(rows)
        rows["count"] = rows.groupby(["question", "answer"])["answer"].transform("size")
        rows["order"] = range(len(rows))
        df = (rows.sort_values(["count", "order"], ascending=[False, True])
                  .drop_duplicates("question")
                  .sort_values("order")
                  .reset_index(drop=True))

        if self.split == "train":
            df = df.sample(frac=self.train_val_ratio, random_state=42)
        elif self.split == "val":
            df = df.drop(df.sample(frac=self.train_val_ratio, random_state=42).index)

        qa_pairs = []
        for _, row in df.iterrows():
            qa_pairs.append({
                "idx": len(qa_pairs),
                "is_mcq": False,
                "question": row["question"],
                "answer": row["answer"]
            })
        return qa_pairs
    
    def _process_medmcqa(self, split=None):

        if split is not None:
            self.split = split
        
        if self.split == "train":
            file_path = os.path.join(self.original_path, "MedMCQA", "data", "train-00000-of-00001.parquet")
        elif self.split in ["val", "test"]:
            file_path = os.path.join(self.original_path, "MedMCQA", "data", "validation-00000-of-00001.parquet")
        else:
            raise ValueError("MedMCQA splits.")
        
        df = pd.read_parquet(file_path)

        if self.split == "train":
            df = df.sample(frac=0.5, random_state=42)

        if self.split == "val":
            df = df.sample(frac=0.5, random_state=42)
        elif self.split == "test":
            not_val_df = df.drop(df.sample(frac=0.5, random_state=42).index)
            df = not_val_df

        qa_pairs = []
        for _, row in df.iterrows():
            try:
                questions = row["question"].strip()
                options = {
                    "A": row["opa"].strip(),
                    "B": row["opb"].strip(),
                    "C": row["opc"].strip(),
                    "D": row["opd"].strip()
                }
                if int(row["cop"]) == 0:
                    answer = "A"
                    answer_text = row["opa"]
                elif int(row["cop"]) == 1:
                    answer = "B"
                    answer_text = row["opb"]
                elif int(row["cop"]) == 2:
                    answer = "C"
                    answer_text = row["opc"]
                else:
                    answer = "D"
                    answer_text = row["opd"]
                qa_pairs.append({
                    "idx": len(qa_pairs),
                    "is_mcq": True,
                    "question": questions,
                    "options": options,
                    "answer": answer,
                    "answer_text": answer_text.strip()
                })
            except Exception as e:
                continue
        return qa_pairs


    def _process_medqa(self, split=None):
        # only use the English US data
        if split is not None:
            self.split = split

        if self.split == "train":
            file_path = os.path.join(self.original_path, "MedQA", "data_clean/questions/US/train.jsonl")
        elif self.split == "val":
            file_path = os.path.join(self.original_path, "MedQA", "data_clean/questions/US/dev.jsonl")
        else:
            file_path = os.path.join(self.original_path, "MedQA", "data_clean/questions/US/test.jsonl")
        
        original_qa_pairs = []
        # read jsonl into list of dicts
        with open(file_path, "r") as f:
            for line in f:
                original_qa_pairs.append(json.loads(line))

        qa_pairs = []
        for item in original_qa_pairs:
            try:
                question = item["question"].strip()
                options = {
                    "A": item["options"]["A"].strip(),
                    "B": item["options"]["B"].strip(),
                    "C": item["options"]["C"].strip(),
                    "D": item["options"]["D"].strip()
                }
                answer = item["answer_idx"].strip()
                answer_text = item["answer"].strip()
                qa_pairs.append({
                    "idx": len(qa_pairs),
                    "is_mcq": True,
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "answer_text": answer_text
                })
            except Exception as e:
                continue

        return qa_pairs


    def _process_medquad(self, split=None):

        if split is not None:
            self.split = split

        file_path = os.path.join(self.original_path, "MedQuAD", "data", "train-00000-of-00001-e36383d177026d53.parquet")
        df = pd.read_parquet(file_path)

        df = df[df['answer'].notna()]

        if self.split in ["train", "val"]:
            raise ValueError("MedQuAD only use for testing.")

        df = df.sample(frac=0.1, random_state=42)

        qa_pairs = []
        for _, row in df.iterrows():
            try:
                question = row["question"]
                answer = row["answer"]
                qa_pairs.append({
                    "idx": len(qa_pairs),
                    "is_mcq": False,
                    "question": question,
                    "answer": answer
                })
            except Exception as e:
                continue
        return qa_pairs


    def _process_medxpertqa(self, split=None):

        if split is not None:
            self.split = split

        if self.split in ["train", "val"]:
            raise ValueError("MedXpertQA only use for testing.")
        
        file_path = os.path.join(self.original_path, "MedXpertQA", "Text", "test.jsonl")
        original_qa_pairs = []
        # read jsonl into list of dicts
        with open(file_path, "r") as f:
            for line in f:
                original_qa_pairs.append(json.loads(line))
        
        qa_pairs = []
        for item in original_qa_pairs:
            try:
                # keep anything before "Answer Choices"
                question = item["question"].split("Answer Choices")[0].strip()
                options = item["options"]
                answer = item["label"].strip()
                answer_text = options[answer].strip()
                qa_pairs.append({
                    "idx": len(qa_pairs),
                    "is_mcq": True,
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "answer_text": answer_text
                })
            except Exception as e:
                continue
        return qa_pairs

    def _process_mmlu(self, split=None):


        if split is not None:
            self.split = split

        if self.split in ["train", "val"]:

            if self.split == "train":
                base_filename = "dev-00000-of-00001.parquet"
            else:
                base_filename = "validation-00000-of-00001.parquet"

            anatomy_path = os.path.join(self.original_path, "MMLU", "anatomy", base_filename)
            clinical_path = os.path.join(self.original_path, "MMLU", "clinical_knowledge", base_filename)
            genetics_path = os.path.join(self.original_path, "MMLU", "medical_genetics", base_filename)
            virology_path = os.path.join(self.original_path, "MMLU", "virology", base_filename)

            df_anatomy = pd.read_parquet(anatomy_path)
            df_clinical = pd.read_parquet(clinical_path)
            df_genetics = pd.read_parquet(genetics_path)
            df_virology = pd.read_parquet(virology_path)
        
            df_list = [df_anatomy, df_clinical, df_genetics, df_virology]

        else:
            base_filename = "test-00000-of-00001.parquet"

            pro_medicine_path = os.path.join(self.original_path, "MMLU", "professional_medicine", base_filename)
            college_medicine_path = os.path.join(self.original_path, "MMLU", "college_medicine", base_filename)
            anatomy_path = os.path.join(self.original_path, "MMLU", "anatomy", base_filename)
            clinical_path = os.path.join(self.original_path, "MMLU", "clinical_knowledge", base_filename)
            genetics_path = os.path.join(self.original_path, "MMLU", "medical_genetics", base_filename)
            virology_path = os.path.join(self.original_path, "MMLU", "virology", base_filename)

            df_pro_medicine = pd.read_parquet(pro_medicine_path)
            df_college_medicine = pd.read_parquet(college_medicine_path)
            df_anatomy = pd.read_parquet(anatomy_path)
            df_clinical = pd.read_parquet(clinical_path)
            df_genetics = pd.read_parquet(genetics_path)
            df_virology = pd.read_parquet(virology_path)

            df_list = [df_pro_medicine,
                        df_college_medicine,
                        df_anatomy,
                        df_clinical,
                        df_genetics,
                        df_virology]

        qa_pairs = []
        for df in df_list:
            for _, row in df.iterrows():
                try:
                    question = row["question"].strip()
                    options_raw = [o.strip() for o in list(row["choices"])]
                    options = {
                        chr(65 + idx): opt for idx, opt in enumerate(options_raw) # A, B, C, D...
                    }
                    
                    answer = chr(int(row["answer"]) + 65)  # convert 0,1,2,3 to ASCII A,B,C,D
                    answer_text = options[answer].strip()
                    qa_pairs.append({
                        "idx": len(qa_pairs),
                        "is_mcq": True,
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "answer_text": answer_text
                    })
                except Exception as e:
                    continue
        return qa_pairs

    def _process_pubmedqa(self, split=None):

        if split is not None:
            self.split = split

        file_path = os.path.join(self.original_path, "PubMedQA", "pqa_labeled", "train-00000-of-00001.parquet")
        df = pd.read_parquet(file_path)

        if self.split == "train":
            df = df.sample(frac=self.train_val_ratio, random_state=42)
        elif self.split == "val":
            not_train_df = df.drop(df.sample(frac=self.train_val_ratio, random_state=42).index)
            # get 50% of not_train_df for val
            df = not_train_df.sample(frac=0.5, random_state=42)
        else:
            not_train_df = df.drop(df.sample(frac=self.train_val_ratio, random_state=42).index)
            # get the other 50% of not_train_df for test
            df = not_train_df.drop(not_train_df.sample(frac=0.5, random_state=42).index)

        qa_pairs = []
        for _, row in df.iterrows():
            try:
                question = row["question"].strip()
                options = {
                    "A": "yes",
                    "B": "no",
                    "C": "maybe"
                }
                answer_text = row["final_decision"].strip()
                if answer_text == "yes":
                    answer = "A"
                elif answer_text == "no":
                    answer = "B"
                else:
                    answer = "C"
                answer_long = row["long_answer"].strip()
                qa_pairs.append(
                    {
                        "idx": len(qa_pairs),
                        "is_mcq": True,
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "answer_text": answer_text,
                        "answer_long": answer_long
                    }
                )
            except Exception as e:
                continue
        return qa_pairs
    

    def make_train_val_test_split(self):

        os.makedirs(os.path.join(self.processed_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_path, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_path, "test"), exist_ok=True)

        metadata = {}
        # =======process training split=======
        medqa_train = self._process_medqa(split="train")
        pubmedqa_train = self._process_pubmedqa(split="train")
        bioasq_train = self._process_bioasq(split="train")
        medmcqa_train = self._process_medmcqa(split="train")

        with open(os.path.join(self.processed_path, "train", "medqa_train.json"), "w") as f:
            json.dump(medqa_train, f, indent=4)
        
        with open(os.path.join(self.processed_path, "train", "pubmedqa_train.json"), "w") as f:
            json.dump(pubmedqa_train, f, indent=4)

        with open(os.path.join(self.processed_path, "train", "bioasq_train.json"), "w") as f:
            json.dump(bioasq_train, f, indent=4)

        with open(os.path.join(self.processed_path, "train", "medmcqa_train.json"), "w") as f:
            json.dump(medmcqa_train, f, indent=4)


        metadata["train"] = {
            "medqa": len(medqa_train),
            "pubmedqa": len(pubmedqa_train),
            "bioasq": len(bioasq_train),
            "medmcqa": len(medmcqa_train),
        }

        # =======process validation split=======
        medqa_val = self._process_medqa(split="val")
        pubmedqa_val = self._process_pubmedqa(split="val")
        bioasq_val = self._process_bioasq(split="val")
        medmcqa_val = self._process_medmcqa(split="val")

        with open(os.path.join(self.processed_path, "val", "medqa_val.json"), "w") as f:
            json.dump(medqa_val, f, indent=4)

        with open(os.path.join(self.processed_path, "val", "pubmedqa_val.json"), "w") as f:
            json.dump(pubmedqa_val, f, indent=4)

        with open(os.path.join(self.processed_path, "val", "bioasq_val.json"), "w") as f:
            json.dump(bioasq_val, f, indent=4)

        with open(os.path.join(self.processed_path, "val", "medmcqa_val.json"), "w") as f:
            json.dump(medmcqa_val, f, indent=4)

        metadata["val"] = {
            "medqa": len(medqa_val),
            "pubmedqa": len(pubmedqa_val),
            "bioasq": len(bioasq_val),
            "medmcqa": len(medmcqa_val),
        }


        # =======process testing split=======
        medqa_test = self._process_medqa(split="test")
        medquad_test = self._process_medquad(split="test")
        bioasq_test = self._process_bioasq(split="test")
        medmcqa_test = self._process_medmcqa(split="test")
        medxpertqa_test = self._process_medxpertqa(split="test")
        mmlu_test = self._process_mmlu(split="test")
        pubmedqa_test = self._process_pubmedqa(split="test")

        with open(os.path.join(self.processed_path, "test", "medqa_test.json"), "w") as f:
            json.dump(medqa_test, f, indent=4)

        with open(os.path.join(self.processed_path, "test", "medquad_test.json"), "w") as f:
            json.dump(medquad_test, f, indent=4)
        
        with open(os.path.join(self.processed_path, "test", "bioasq_test.json"), "w") as f:
            json.dump(bioasq_test, f, indent=4)
        
        with open(os.path.join(self.processed_path, "test", "medmcqa_test.json"), "w") as f:
            json.dump(medmcqa_test, f, indent=4)

        with open(os.path.join(self.processed_path, "test", "medxpertqa_test.json"), "w") as f:
            json.dump(medxpertqa_test, f, indent=4)

        with open(os.path.join(self.processed_path, "test", "mmlu_test.json"), "w") as f:
            json.dump(mmlu_test, f, indent=4)

        with open(os.path.join(self.processed_path, "test", "pubmedqa_test.json"), "w") as f:
            json.dump(pubmedqa_test, f, indent=4)

        metadata["test"] = {
            "medqa": len(medqa_test),
            "medquad": len(medquad_test),
            "bioasq": len(bioasq_test),
            "medmcqa": len(medmcqa_test),
            "medxpertqa": len(medxpertqa_test),
            "mmlu": len(mmlu_test),
            "pubmedqa": len(pubmedqa_test),
        }

        with open(os.path.join(self.processed_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)


    def _helper_generate_trajectories(self, model_info, dict_processed_qa_pair, num_trajectories=100, batch_size=100):

        if dict_processed_qa_pair["is_mcq"]:
            system_prompt = mcq_system_prompt
            options_formatted = "\n".join([f"{key}: {value}" for key, value in dict_processed_qa_pair["options"].items()])
            user_prompt = str(dict_processed_qa_pair["question"]) + "\n" + "Options:\n" + options_formatted
        else:
            system_prompt = frq_system_prompt
            user_prompt = str(dict_processed_qa_pair["question"])

        responses = []
        for batch_start in range(0, num_trajectories, batch_size):
            current_batch_size = min(batch_size, num_trajectories - batch_start)
            system_prompts = [system_prompt] * current_batch_size
            user_prompts = [user_prompt] * current_batch_size

            batch_responses = get_response(
                model_info=model_info,
                system_prompts=system_prompts,
                user_prompts=user_prompts,
                has_thinking=False
            )

            responses.extend(batch_responses)
        return responses


    def collect_trajectories(self, num_trajectories=100, batch_size=100):
        
        if os.path.exists(self.processed_path) is False:
            
            print("Processed data not found. ")
            print("Creating train val test split...")
            # run make_train_val_test_split to process datasets
            self.make_train_val_test_split()
        
        model_info = setup_model(self.model_name, use_vllm=self.use_vllm)

        qa_pair_path = os.path.join(self.processed_path, self.split, f"{self.dataset_name}_{self.split}.json")

        with open(qa_pair_path, "r") as f:
            qa_pairs = json.load(f)

        sampled_trajectories_path = os.path.join(self.samples_path, self.split, f"{self.dataset_name}_{self.split}", self.model_name.replace("/", "_"))
        os.makedirs(sampled_trajectories_path, exist_ok=True)

        # shuffle qa_pairs
        import random
        random.shuffle(qa_pairs)
        for qa_pair in qa_pairs:
            idx = qa_pair["idx"]
            print(idx)
            trajectory_file_path = os.path.join(sampled_trajectories_path, f"qa_{idx}_trajectories.jsonl")
            if os.path.exists(trajectory_file_path):
                print(f"Trajectories for QA pair {idx} already exist. Skipping...")
                continue
            print(f"Generating trajectories for QA pair {idx}...")
            trajectories = self._helper_generate_trajectories(model_info, qa_pair, num_trajectories=num_trajectories, batch_size=batch_size)
            with open(trajectory_file_path, "w") as f:
                for traj in trajectories:
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, required=True, help="Model name or path")
    args.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    args.add_argument("--split", type=str, required=True, help="Data split: train/val/test")
    args.add_argument("--original_path", type=str, default="./data/original", help="Path to original data")
    args.add_argument("--processed_path", type=str, default="./data/processed", help="Path to processed data")
    args.add_argument("--samples_path", type=str, default="./data/samples", help="Path to save sampled trajectories")
    args.add_argument("--train_val_ratio", type=float, default=0.8, help="Train/validation split ratio")
    args.add_argument("--use_vllm", action="store_true", help="Whether to use vLLM for inference")
    args.add_argument("--num_trajectories", type=int, default=100, help="Number of trajectories to sample per QA pair")
    args.add_argument("--batch_size", type=int, default=10, help="Batch size for trajectory generation")

    parsed_args = args.parse_args()
    collector = TrajectoryCollector(
        model_name=parsed_args.model_name,
        dataset_name=parsed_args.dataset_name,
        split=parsed_args.split,
        original_path=parsed_args.original_path,
        processed_path=parsed_args.processed_path,
        samples_path=parsed_args.samples_path,
        train_val_ratio=parsed_args.train_val_ratio,
        use_vllm=parsed_args.use_vllm
    )
    
    collector.collect_trajectories(
        num_trajectories=parsed_args.num_trajectories,
        batch_size=parsed_args.batch_size
    )



