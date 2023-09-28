"use client";
import React, { useEffect } from "react";

const Error = ({ error, reset }: { error: Error; reset: () => void }) => {
  useEffect(() => {
    console.log("Error: ", error.message);
    return () => {
      reset();
    };
  }, [error]);

  return (
    <>
      <div>{error.message}</div>
      <button onClick={() => reset()}>Try again</button>
    </>
  );
};

export default Error;
