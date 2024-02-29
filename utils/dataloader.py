import torch


class BalancedDataLoaderIterator:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

        self.num_dataloaders = len(dataloaders)

        max_length = max(len(dataloader) for dataloader in dataloaders)

        length_list = [len(dataloader) for dataloader in dataloaders]
        print("data loader length:", length_list)
        print("max dataloader length:", max_length,
              "epoch iteration:", max_length * self.num_dataloaders)
        self.total_length = max_length * self.num_dataloaders
        self.current_iteration = 0
        self.probabilities = torch.ones(
            self.num_dataloaders, dtype=torch.float) / self.num_dataloaders

    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.current_iteration = 0
        return self

    def __next__(self):
        if self.current_iteration >= self.total_length:
            raise StopIteration

        chosen_index = torch.multinomial(self.probabilities, 1).item()
        try:
            sample = next(self.iterators[chosen_index])
        except StopIteration:
            self.iterators[chosen_index] = iter(self.dataloaders[chosen_index])
            sample = next(self.iterators[chosen_index])

        self.current_iteration += 1
        return sample, chosen_index

    def __len__(self):
        return self.total_length

    def generate_fake_samples_for_batch(self, dataloader_id, batch_size):
        if dataloader_id >= len(self.dataloaders) or dataloader_id < 0:
            raise ValueError("Invalid dataloader ID")

        dataloader = self.dataloaders[dataloader_id]
        iterator = iter(dataloader)

        try:
            sample_batch = next(iterator)
            fake_samples = []

            for sample in sample_batch:
                if isinstance(sample, torch.Tensor):
                    fake_sample = torch.zeros(
                        [batch_size] + list(sample.shape)[1:])
                    fake_samples.append(fake_sample)
                else:
                    pass

            return fake_samples, dataloader_id
        except StopIteration:
            return None
