import great_expectations
import sys

def main():

    data_context = great_expectations.get_context()

    validate_data = data_context.run_checkpoint(
        checkpoint_name="my_checkpoint",
        batch_request=None,
        run_name=None
    )

    if not validate_data["success"]:
        print("Validation failed!")
        sys.exit(1)
    else:
        print("Validation succeeded!")

if __name__ == '__main__':
    main()
