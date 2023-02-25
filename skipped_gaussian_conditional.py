class SkippedGaussianConditional(GaussianConditional):
    def __init__(
        self,
        *args: Any,
        tau: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tau = float(tau)
       
    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(inputs, "skip-noise" if training else "skip-dequantize", means, scales)
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood 
    
    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        likelihood = torch.where(scales<self.tau, likelihood.detach(), likelihood)

        return likelihood

    def compress(self, inputs, indexes, means=None, scales=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, "symbols", means)

        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            non_skip = ((scales[i].reshape(-1)) >= self.tau)
            symbols_i = (symbols[i].reshape(-1))[non_skip]
            indexes_i = (indexes[i].reshape(-1))[non_skip]
            
            if (len(symbols_i) == 0):
                strings.append("")
                continue
            
            rv = self.entropy_coder.encode_with_indexes(
                symbols_i.int().tolist(),
                indexes_i.int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
        scales: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """
        if strings == []:
            strings = [""] * (indexes.size(0))

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            if s == "":
                values = torch.zeros(indexes[i].size())
                outputs[i] = torch.tensor(
                    values, device=outputs.device, dtype=outputs.dtype
                ).reshape(outputs[i].size())
                continue

            non_skip = ((scales[i].reshape(-1)) >= self.tau)
            indexes_i = (indexes[i].reshape(-1))[non_skip]
            
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes_i.int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            
            output = torch.zeros_like((outputs[i].reshape(-1)))
            newi = 0
            for pos in non_skip.nonzero():
                output[pos] = values[newi]
                newi += 1
            outputs[i] = output.reshape(outputs[i].size())

        outputs = outputs + means
        return outputs
    
    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None, scales: Optional[Tensor] = None,
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols", "skip-noise", "skip-dequantize"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')
        if scales != None:
            assert scales.shape == inputs.shape

        if mode == "noise" or mode == "skip-noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            if mode == "noise": 
                return inputs
            if mode == "skip-noise":
                assert scales != None
                return torch.where(scales<self.tau, means, inputs)

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize" or mode == "skip-dequantize":
            if means is not None:
                outputs += means
            if mode == "dequantize":
                return outputs
            if mode == "skip-dequantize":
                assert scales != None
                return torch.where(scales<self.tau, means, outputs)

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs
