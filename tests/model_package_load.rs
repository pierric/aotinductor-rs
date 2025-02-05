#[cfg(test)]
mod tests {
    use aotinductor::ModelPackage;
    use tch::Tensor;

    #[test]
    fn test_model_package_load_bad() {
        let result = ModelPackage::new("non-existent-file");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_package_load_good() {
        let result = ModelPackage::new("tests/ep1.pt2");
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn test_model_package_run1() {
        let result = ModelPackage::new("tests/ep1.pt2");
        assert!(result.is_ok());
        let loader = result.unwrap();
        let inp = Tensor::rand([1, 32], (tch::Kind::Float, tch::Device::Cpu));
        let out = loader.run(&vec![inp]);
        assert!(out.len() == 1, "outputed {} tensors, expecting 1", out.len());
        assert!(out[0].size() == [1], "output tensor shape {:?}, expecting [1]", out[0].size())
    }

    #[test]
    fn test_model_package_run2() {
        let result = ModelPackage::new("tests/ep2.pt2");
        assert!(result.is_ok());
        let loader = result.unwrap();
        let inp1 = Tensor::rand([1, 2], (tch::Kind::Float, tch::Device::Cpu));
        let inp2 = Tensor::rand([1, 4], (tch::Kind::Float, tch::Device::Cpu));
        let out = loader.run(&vec![inp1, inp2]);
        assert!(out.len() == 2, "output {} tensors, expecting 2", out.len());
    }
}
